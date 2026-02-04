"""
Backend Client implementation for communication with the QuackOps web server.

This module provides WebSocket and HTTP communication with the
drone control backend (qpsDroneApiService).
"""

import asyncio
import json
import logging
import time
from typing import Optional, Callable, Any, Dict
from dataclasses import asdict
from enum import Enum

from ..core.interfaces import (
    IBackendClient,
    DroneTelemetry,
    LandingPhase,
    DroneState,
)
from ..core.config import CommunicationConfig

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of WebSocket messages."""
    TELEMETRY = "telemetry"
    LANDING_STATUS = "landing_status"
    COMMAND = "command"
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    ERROR = "error"


class BackendClient(IBackendClient):
    """
    Client for communicating with the QuackOps web backend.
    
    Handles both WebSocket communication for real-time telemetry
    and HTTP requests for commands and status updates.
    
    Attributes:
        config: Communication configuration settings
        
    Example:
        >>> client = BackendClient(config)
        >>> await client.connect()
        >>> await client.send_telemetry(telemetry)
        >>> await client.disconnect()
    """
    
    def __init__(self, config: CommunicationConfig):
        """
        Initialize the backend client.
        
        Args:
            config: Communication configuration settings
        """
        self.config = config
        self._websocket = None
        self._is_connected = False
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        
        # Command callbacks
        self._command_callbacks: list = []
        
        # Message queue for outgoing messages
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._send_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._messages_sent = 0
        self._messages_received = 0
        self._last_heartbeat_time = 0.0
        
        logger.info(
            f"BackendClient initialized (ws={config.websocket_url})"
        )
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection to the backend server.
        
        Returns:
            True if connection established successfully
        """
        if self._is_connected:
            logger.warning("Already connected to backend")
            return True
        
        try:
            import websockets
            
            logger.info(f"Connecting to {self.config.websocket_url}...")
            
            self._websocket = await websockets.connect(
                self.config.websocket_url,
                ping_interval=20,
                ping_timeout=10,
            )
            
            self._is_connected = True
            logger.info("Connected to backend server")
            
            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._send_task = asyncio.create_task(self._send_loop())
            
            # Send initial connection message
            await self._send_message({
                "type": "connect",
                "client": "pi_cv_module",
                "timestamp": time.time()
            })
            
            return True
            
        except ImportError:
            logger.error("websockets not installed. Install with: pip install websockets")
            return False
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self._is_connected = False
            
            # Start reconnect task
            if self._reconnect_task is None or self._reconnect_task.done():
                self._reconnect_task = asyncio.create_task(self._reconnect_loop())
            
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the backend server."""
        self._is_connected = False
        
        # Cancel background tasks
        for task in [
            self._heartbeat_task,
            self._receive_task,
            self._send_task,
            self._reconnect_task
        ]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close WebSocket
        if self._websocket is not None:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        self._websocket = None
        logger.info("Disconnected from backend server")
    
    async def send_telemetry(self, telemetry: DroneTelemetry) -> bool:
        """
        Send telemetry data to the backend.
        
        Args:
            telemetry: Current drone telemetry data
            
        Returns:
            True if message queued successfully
        """
        if not self._is_connected:
            return False
        
        message = {
            "type": MessageType.TELEMETRY.value,
            "timestamp": time.time(),
            "data": {
                "position": {
                    "latitude": telemetry.position.latitude,
                    "longitude": telemetry.position.longitude,
                    "altitude_m": telemetry.position.altitude_m,
                    "relative_altitude_m": telemetry.position.relative_altitude_m,
                    "heading_deg": telemetry.position.heading_deg,
                },
                "velocity": {
                    "north_mps": telemetry.velocity.velocity_north_mps,
                    "east_mps": telemetry.velocity.velocity_east_mps,
                    "down_mps": telemetry.velocity.velocity_down_mps,
                    "groundspeed_mps": telemetry.velocity.groundspeed_mps,
                },
                "battery_percent": telemetry.battery_percent,
                "is_armed": telemetry.is_armed,
                "is_in_air": telemetry.is_in_air,
                "flight_mode": telemetry.flight_mode,
                "gps_fix_type": telemetry.gps_fix_type,
                "satellite_count": telemetry.satellite_count,
                "state": telemetry.state.name,
            }
        }
        
        await self._queue_message(message)
        return True
    
    async def send_landing_status(
        self,
        phase: LandingPhase,
        marker_detected: bool,
        details: Optional[dict] = None
    ) -> bool:
        """
        Send landing status update to the backend.
        
        Args:
            phase: Current landing phase
            marker_detected: Whether the target marker is currently detected
            details: Optional additional details
            
        Returns:
            True if message queued successfully
        """
        if not self._is_connected:
            return False
        
        message = {
            "type": MessageType.LANDING_STATUS.value,
            "timestamp": time.time(),
            "data": {
                "phase": phase.name,
                "marker_detected": marker_detected,
                "details": details or {}
            }
        }
        
        await self._queue_message(message)
        return True
    
    async def send_event(
        self,
        event_type: str,
        data: Optional[dict] = None
    ) -> bool:
        """
        Send a generic event to the backend.
        
        Args:
            event_type: Type of event
            data: Event data
            
        Returns:
            True if message queued successfully
        """
        if not self._is_connected:
            return False
        
        message = {
            "type": "event",
            "event_type": event_type,
            "timestamp": time.time(),
            "data": data or {}
        }
        
        await self._queue_message(message)
        return True
    
    def register_command_callback(
        self,
        callback: Callable[[str, dict], Any]
    ) -> None:
        """
        Register callback for commands received from backend.
        
        Args:
            callback: Function to call with (command_type, command_data)
        """
        self._command_callbacks.append(callback)
        logger.debug(f"Registered command callback: {callback}")
    
    async def _queue_message(self, message: dict) -> None:
        """Add message to the send queue."""
        await self._message_queue.put(message)
    
    async def _send_message(self, message: dict) -> bool:
        """Send a message immediately."""
        if not self._is_connected or self._websocket is None:
            return False
        
        try:
            await self._websocket.send(json.dumps(message))
            self._messages_sent += 1
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self._is_connected = False
            return False
    
    async def _send_loop(self) -> None:
        """Background task to send queued messages."""
        while self._is_connected:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                await self._send_message(message)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send loop error: {e}")
    
    async def _receive_loop(self) -> None:
        """Background task to receive messages from backend."""
        while self._is_connected and self._websocket is not None:
            try:
                message_str = await self._websocket.recv()
                message = json.loads(message_str)
                self._messages_received += 1
                
                await self._handle_message(message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Receive loop error: {e}")
                self._is_connected = False
                break
    
    async def _handle_message(self, message: dict) -> None:
        """Handle incoming message from backend."""
        msg_type = message.get("type", "")
        
        if msg_type == MessageType.COMMAND.value:
            command = message.get("command", "")
            data = message.get("data", {})
            
            logger.info(f"Received command: {command}")
            
            # Notify all registered callbacks
            for callback in self._command_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(command, data)
                    else:
                        callback(command, data)
                except Exception as e:
                    logger.error(f"Command callback error: {e}")
            
            # Send acknowledgment
            await self._send_message({
                "type": MessageType.ACK.value,
                "command": command,
                "timestamp": time.time()
            })
        
        elif msg_type == MessageType.HEARTBEAT.value:
            # Respond to heartbeat
            await self._send_message({
                "type": MessageType.HEARTBEAT.value,
                "timestamp": time.time()
            })
        
        elif msg_type == MessageType.ACK.value:
            logger.debug(f"Received ACK for: {message.get('command', 'unknown')}")
    
    async def _heartbeat_loop(self) -> None:
        """Background task to send periodic heartbeats."""
        while self._is_connected:
            try:
                await asyncio.sleep(self.config.heartbeat_interval_sec)
                
                await self._send_message({
                    "type": MessageType.HEARTBEAT.value,
                    "timestamp": time.time()
                })
                
                self._last_heartbeat_time = time.time()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _reconnect_loop(self) -> None:
        """Background task to attempt reconnection."""
        while not self._is_connected:
            try:
                await asyncio.sleep(self.config.reconnect_interval_sec)
                logger.info("Attempting to reconnect...")
                
                if await self.connect():
                    break
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to backend."""
        return self._is_connected
    
    @property
    def statistics(self) -> dict:
        """Get connection statistics."""
        return {
            "is_connected": self._is_connected,
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "last_heartbeat": self._last_heartbeat_time,
        }


class HttpBackendClient:
    """
    HTTP client for REST API communication with backend.
    
    Used for non-real-time operations like fetching mission data,
    updating order status, etc.
    """
    
    def __init__(self, config: CommunicationConfig):
        """
        Initialize HTTP client.
        
        Args:
            config: Communication configuration settings
        """
        self.config = config
        self.base_url = config.backend_base_url.rstrip("/")
        
    async def get(self, endpoint: str) -> Optional[dict]:
        """Make GET request to backend."""
        try:
            import aiohttp
            
            url = f"{self.base_url}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.config.api_timeout_sec)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"GET {endpoint} failed: {response.status}")
                        return None
                        
        except ImportError:
            logger.error("aiohttp not installed. Install with: pip install aiohttp")
            return None
        except Exception as e:
            logger.error(f"GET request failed: {e}")
            return None
    
    async def post(
        self, 
        endpoint: str, 
        data: Optional[dict] = None
    ) -> Optional[dict]:
        """Make POST request to backend."""
        try:
            import aiohttp
            
            url = f"{self.base_url}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self.config.api_timeout_sec)
                ) as response:
                    if response.status in (200, 201):
                        return await response.json()
                    else:
                        logger.error(f"POST {endpoint} failed: {response.status}")
                        return None
                        
        except ImportError:
            logger.error("aiohttp not installed")
            return None
        except Exception as e:
            logger.error(f"POST request failed: {e}")
            return None
    
    async def get_pending_orders(self) -> list:
        """Fetch pending delivery orders."""
        result = await self.get("/api/orders/pending")
        return result.get("orders", []) if result else []
    
    async def update_order_status(
        self, 
        order_id: str, 
        status: str, 
        details: Optional[dict] = None
    ) -> bool:
        """Update an order's delivery status."""
        data = {
            "status": status,
            "updated_at": time.time(),
            "details": details or {}
        }
        result = await self.post(f"/api/orders/{order_id}/status", data)
        return result is not None
    
    async def report_landing_complete(
        self, 
        order_id: str,
        landing_accuracy_cm: float
    ) -> bool:
        """Report successful landing completion."""
        data = {
            "order_id": order_id,
            "landing_accuracy_cm": landing_accuracy_cm,
            "completed_at": time.time()
        }
        result = await self.post("/api/drone/landing-complete", data)
        return result is not None
