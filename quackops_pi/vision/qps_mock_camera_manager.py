from __future__ import annotations

from typing import Optional

import numpy

from quackops_pi.config.qps_config import qpsConfig
from quackops_pi.vision.qps_camera_manager_interface import qpsCameraManagerInterface


class qpsMockCameraManager(qpsCameraManagerInterface):
    """Mock camera manager that serves pre-loaded test frames.

    Cycles through a list of frames loaded via load_test_frames().
    Set fail_on_start=True to simulate a camera that fails to initialise.
    Set return_none=True to simulate a camera that returns None frames.
    """

    def __init__(self, config: qpsConfig) -> None:
        self.config: qpsConfig = config
        self.frames: list[numpy.ndarray] = []
        self.frame_index: int = 0
        self.running: bool = False
        self.fail_on_start: bool = False
        self.return_none: bool = False

    async def start(self) -> None:
        if self.fail_on_start:
            raise RuntimeError("Mock camera: simulated start failure")
        self.running = True
        self.frame_index = 0

    async def stop(self) -> None:
        self.running = False

    async def get_frame(self) -> Optional[numpy.ndarray]:
        if self.return_none:
            return None
        if not self.frames:
            return numpy.zeros((480, 640, 3), dtype=numpy.uint8)
        frame = self.frames[self.frame_index % len(self.frames)]
        self.frame_index += 1
        return frame

    def is_running(self) -> bool:
        return self.running

    # ── Test-helper methods ───────────────────────────────────────────────────

    def load_test_frames(self, frames: list[numpy.ndarray]) -> None:
        self.frames = list(frames)
        self.frame_index = 0
