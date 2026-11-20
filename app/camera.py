"""Camera capture: Picamera2 (Pi Camera Module 3) -> OpenCV (USB webcam) -> none (upload only)."""
import threading

import numpy as np


class Camera:
    def __init__(self, prefer="auto", index=0, resolution=(1280, 960)):
        self.kind = None
        self._lock = threading.Lock()
        if prefer in ("auto", "picamera2"):
            try:
                from picamera2 import Picamera2
                self.cam = Picamera2()
                self.cam.configure(self.cam.create_still_configuration(
                    main={"size": resolution, "format": "RGB888"}))
                self.cam.start()
                self.kind = "picamera2"
                return
            except Exception as e:
                if prefer == "picamera2":
                    raise
                print(f"[camera] Picamera2 not available ({e.__class__.__name__}); trying OpenCV")
        if prefer in ("auto", "opencv"):
            try:
                import cv2  # noqa: F401
                self.index, self.resolution = index, resolution
                cap = self._open_cv()
                if cap.isOpened():
                    # Opened per capture and released again: keeping the legacy Pi camera
                    # streaming all day wastes CPU, and a stuck stream never recovers.
                    cap.release()
                    self.cam, self.kind = None, "opencv"
                    return
                cap.release()
            except ImportError:
                pass
        print("[camera] no camera found - upload-only mode")

    def _open_cv(self):
        import sys
        import cv2
        if sys.platform.startswith("linux"):
            cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
            # The Pi Camera in legacy mode (bcm2835-v4l2) defaults to BGR3, which OpenCV
            # returns as a flat (1, W*H*3) buffer. MJPG gives a proper HxWx3 frame.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        else:
            cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        return cap

    def _capture_cv(self):
        import cv2
        cap = self._open_cv()
        try:
            for _ in range(5):  # let auto-exposure / white balance settle
                cap.grab()
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            raise RuntimeError("camera read failed (Pi camera: check the ribbon cable; "
                               "dmesg 'Failed enabling camera' = power-cycle the Pi)")
        if frame.ndim != 3:  # flat raw buffer: reshape using the requested size
            w, h = self.resolution
            frame = frame.reshape(h, w, 3)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @property
    def available(self):
        return self.kind is not None

    def capture(self):
        """Return an HxWx3 RGB uint8 array."""
        with self._lock:
            if self.kind == "picamera2":
                # Picamera2's "RGB888" is BGR byte order in memory -> flip to RGB
                return np.ascontiguousarray(self.cam.capture_array("main")[:, :, ::-1])
            if self.kind == "opencv":
                try:
                    return self._capture_cv()
                except RuntimeError:
                    return self._capture_cv()  # one retry: the first open after idle can fail
            raise RuntimeError("no camera")

    def close(self):
        if self.kind == "picamera2":
            self.cam.stop()
        # opencv: nothing held open between captures
