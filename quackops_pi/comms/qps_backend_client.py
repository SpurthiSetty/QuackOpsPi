"""
qps_backend_client.py

Production WebSocket client for bidirectional communication with the Node.js backend.

Uses the `websockets` library (same pattern as _WSBridge in test_fly_to_dest.py).
Maintains a persistent connection, queues outbound messages, and dispatches
incoming startDelivery commands to registered callbacks. Reconnects automatically
on connection loss.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

import websockets

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.comms.qps_backend_client_interface import qpsBackendClientInterface
from quackops_pi.models.qps_gps_position import qpsGPSPosition
from quackops_pi.models.qps_mission_command import qpsMissionCommand
from quackops_pi.models.qps_command_type import qpsCommandType

logger = logging.getLogger("qps.backend_client")


class qpsBackendClient(qpsBackendClientInterface):
    """Production WebSocket client for the Node.js backend.

    Architecture:
        connect() opens the WebSocket and starts _recv_loop + _send_loop tasks.
        Outbound messages are enqueued via send_status() / stream_gps() and
        drained by _send_loop.  Inbound startDelivery messages are parsed and
        forwarded to the registered command callback.  On connection loss,
        _reconnect_loop retries at config.reconnection_interval_s intervals.
        Messages queued while disconnected are flushed after reconnection.
    """

    def __init__(self, config: qpsConfig) -> None:
        self._config = config
        self._ws: Optional[Any] = None          # websockets.WebSocketClientProtocol
        self._connected: bool = False

        self._command_callback: Optional[Callable] = None
        self._return_to_source_callback: Optional[Callable] = None

        # Outbound queue consumed by _send_loop
        self._send_queue: asyncio.Queue = asyncio.Queue()

        # Messages buffered while disconnected; flushed on reconnection
        self._pending: List[str] = []

        self._recv_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

    # ── Connection ────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open WebSocket to backend and start recv/send tasks.

        On failure, starts _reconnect_loop in the background so the
        MissionController doesn't have to handle connection retries.
        """
        url = self._config.backend_ws_url + "?role=pi"
        logger.info("Connecting to backend at %s", url)
        try:
            self._ws = await websockets.connect(url)
            self._connected = True
            logger.info("Backend WebSocket connected")
            self._recv_task = asyncio.create_task(
                self._recv_loop(), name="backend-recv"
            )
            self._send_task = asyncio.create_task(
                self._send_loop(), name="backend-send"
            )
            await self._flush_pending()
        except Exception as exc:
            logger.warning("Backend connect failed: %s — will retry", exc)
            self._connected = False
            self._reconnect_task = asyncio.create_task(
                self._reconnect_loop(), name="backend-reconnect"
            )

    async def disconnect(self) -> None:
        """Close the WebSocket and cancel all background tasks."""
        self._connected = False
        for task in (self._recv_task, self._send_task, self._reconnect_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._recv_task = None
        self._send_task = None
        self._reconnect_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        logger.info("Backend client disconnected")

    # ── Outbound API ──────────────────────────────────────────────────

    async def send_status(self, status_type: str, data: Dict[str, Any]) -> None:
        """Serialize and enqueue a status message.

        Format: {"type": "status", "status_type": <str>, "data": <dict>}
        Exception: DELIVERY_COMPLETE sends {"type": "delivered", "orderId": <order_id>}.
        If not connected, the message is buffered for delivery after reconnection.
        """
        if status_type == "DELIVERY_COMPLETE":
            payload = json.dumps({
                "type": "delivered",
                "orderId": data["order_id"],
            })
        elif status_type == "MISSION_COMPLETE":
            payload = json.dumps({
                "type": "fulfilled",
                "orderId": data["order_id"],
            })
        else:
            payload = json.dumps({
                "type": "status",
                "status_type": status_type,
                "data": data,
            })
        if self._connected:
            await self._send_queue.put(payload)
        else:
            self._pending.append(payload)
            logger.debug("Queued status (offline): %s", status_type)

    async def stream_gps(self, position: qpsGPSPosition) -> None:
        """Enqueue a GPS position update.

        Format: {"position": {"latitude_deg": ..., "longitude_deg": ..., "altitude_m": ...}}
        Dropped silently if not connected (GPS stream is best-effort).
        """
        if not self._connected:
            return
        payload = json.dumps({
            "position": {
                "latitude_deg": position.latitude_deg,
                "longitude_deg": position.longitude_deg,
                "altitude_m": position.altitude_m,
            }
        })
        # Non-blocking: drop GPS if queue is backed up to avoid memory growth
        try:
            self._send_queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    # ── Inbound API ───────────────────────────────────────────────────

    def on_command(self, callback: Callable) -> None:
        """Register a callback that receives qpsMissionCommand on incoming commands."""
        self._command_callback = callback

    def on_return_to_source(self, callback: Callable) -> None:
        """Register a callback invoked when the backend sends returnToSource."""
        self._return_to_source_callback = callback

    def is_connected(self) -> bool:
        """Return True if the WebSocket connection is currently active."""
        return self._connected

    # ── Internal loops ────────────────────────────────────────────────

    async def _recv_loop(self) -> None:
        """Consume incoming WebSocket messages and dispatch commands."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Backend sent non-JSON: %r", raw)
                    continue

                msg_type = msg.get("type")
                if msg_type == "startDelivery":
                    self._handle_start_delivery(msg)
                elif msg_type == "returnToSource":
                    self._handle_return_to_source(msg)
                else:
                    logger.info("Backend message: %s", msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Backend recv loop ended: %s", exc)
            await self._on_connection_lost()

    async def _send_loop(self) -> None:
        """Drain the outbound queue and send messages over WebSocket."""
        try:
            while True:
                payload = await self._send_queue.get()
                if not self._connected or self._ws is None:
                    self._pending.append(payload)
                    continue
                try:
                    await self._ws.send(payload)
                except Exception as exc:
                    logger.warning("Backend send error: %s", exc)
                    self._pending.append(payload)
                    await self._on_connection_lost()
        except asyncio.CancelledError:
            raise

    async def _reconnect_loop(self) -> None:
        """Attempt to reconnect to the backend on connection loss."""
        url = self._config.backend_ws_url + "?role=pi"
        while not self._connected:
            logger.info(
                "Reconnecting to backend in %.1fs...",
                self._config.reconnection_interval_s,
            )
            await asyncio.sleep(self._config.reconnection_interval_s)
            try:
                self._ws = await websockets.connect(url)
                self._connected = True
                logger.info("Backend reconnected")
                # Restart recv/send tasks if they've exited
                if self._recv_task is None or self._recv_task.done():
                    self._recv_task = asyncio.create_task(
                        self._recv_loop(), name="backend-recv"
                    )
                if self._send_task is None or self._send_task.done():
                    self._send_task = asyncio.create_task(
                        self._send_loop(), name="backend-send"
                    )
                await self._flush_pending()
                return
            except Exception as exc:
                logger.warning("Reconnect attempt failed: %s", exc)

    async def _flush_pending(self) -> None:
        """Send any messages buffered while disconnected."""
        if not self._pending:
            return
        logger.info("Flushing %d pending messages", len(self._pending))
        flushed: List[str] = []
        for payload in self._pending:
            try:
                await self._ws.send(payload)
                flushed.append(payload)
            except Exception as exc:
                logger.warning("Flush error (will retry after next reconnect): %s", exc)
                break
        for p in flushed:
            self._pending.remove(p)

    async def _on_connection_lost(self) -> None:
        """Handle an unexpected connection drop."""
        if self._connected:
            self._connected = False
            logger.warning("Backend connection lost — starting reconnect loop")
            if self._reconnect_task is None or self._reconnect_task.done():
                self._reconnect_task = asyncio.create_task(
                    self._reconnect_loop(), name="backend-reconnect"
                )

    # ── Message handlers ──────────────────────────────────────────────

    def _handle_start_delivery(self, msg: dict) -> None:
        """Parse startDelivery and invoke the registered command callback."""
        try:
            command = qpsMissionCommand(
                command_type=qpsCommandType.DISPATCH,
                order_id=str(msg["orderId"]),
                destination_lat=float(msg["target"]["lat"]),
                destination_lon=float(msg["target"]["lng"]),
                delivery_marker_id=int(msg.get("markerId", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("Malformed startDelivery message: %s — %s", msg, exc)
            return

        logger.info(
            "startDelivery: orderId=%s  dest=%.6f,%.6f  markerId=%d",
            command.order_id,
            command.destination_lat,
            command.destination_lon,
            command.delivery_marker_id,
        )
        if self._command_callback is not None:
            try:
                self._command_callback(command)
            except Exception as exc:
                logger.warning("Command callback error: %s", exc)

    def _handle_return_to_source(self, msg: dict) -> None:
        """Parse returnToSource and invoke the registered callback."""
        order_id = msg.get("orderId", "")
        logger.info("returnToSource received: orderId=%s", order_id)
        if self._return_to_source_callback is not None:
            try:
                self._return_to_source_callback(order_id)
            except Exception as exc:
                logger.warning("Return-to-source callback error: %s", exc)
