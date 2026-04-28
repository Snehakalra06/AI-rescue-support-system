# main.py — Phase 1 Entry Point
import cv2
import csv
import os
import sys
import time
import threading
from datetime import datetime

import config
from detector import SurvivorDetector
from alert    import trigger_alert


class StreamReader:
    def __init__(self, stream_url, max_retries: int = 5, retry_delay: int = 3):
        self.stream_url  = stream_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._cap        = None
        self._frame      = None
        self._running    = False
        self._lock       = threading.Lock()
        self._thread     = None

    def start(self) -> bool:
        if not self._connect():
            return False
        self._running = True
        self._thread  = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        print(f"[Stream] ✅ Streaming from: {self.stream_url}")
        return True

    def _connect(self) -> bool:
        for attempt in range(1, self.max_retries + 1):
            print(f"[Stream] Connecting... (Attempt {attempt}/{self.max_retries})")

            # Use CAP_DSHOW for webcam (fixes MSMF -1072875772 on Windows)
            if isinstance(self.stream_url, int):
                self._cap = cv2.VideoCapture(self.stream_url, cv2.CAP_DSHOW)
            else:
                self._cap = cv2.VideoCapture(self.stream_url)

            if self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame
                    print(f"[Stream] ✅ Connected successfully.")
                    return True

            self._cap.release()
            print(f"[Stream] ⚠️  Connection failed. Retrying in {self.retry_delay}s...")
            time.sleep(self.retry_delay)

        print(f"[Stream] ❌ Could not connect after {self.max_retries} attempts.")
        return False

    def _update(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                print("[Stream] ⚠️  Frame read failed. Attempting reconnect...")
                self._cap.release()
                if not self._connect():
                    print("[Stream] ❌ Reconnection failed. Stopping stream.")
                    self._running = False
                    break
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_running(self) -> bool:
        return self._running

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        print("[Stream] Stream stopped.")


class SessionLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"mission_{ts}.csv")
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "frame_number", "survivor_count",
                "confidences", "screenshot_path",
            ])
        print(f"[Logger] 📋 Session log: {self.path}")

    def log(self, timestamp, frame_num, survivor_count, confidences, screenshot_path):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, frame_num, survivor_count,
                str([round(c, 2) for c in confidences]),
                screenshot_path,
            ])


class FPSTracker:
    def __init__(self, window: int = 30):
        self._times  = []
        self._window = window

    def tick(self):
        self._times.append(time.time())
        if len(self._times) > self._window:
            self._times.pop(0)

    def get_fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return round((len(self._times) - 1) / elapsed, 1) if elapsed > 0 else 0.0


def draw_fps(frame, fps: float):
    h, w = frame.shape[:2]
    cv2.putText(
        frame, f"FPS: {fps}", (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def draw_paused(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (w//4, h//2 - 30), (3*w//4, h//2 + 30), (0, 0, 0), -1)
    cv2.putText(
        frame, "PAUSED  —  Press SPACE to Resume",
        (w//4 + 10, h//2 + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return frame


def main():
    print("=" * 52)
    print("   AI RESCUE DRONE — SURVIVOR DETECTION SYSTEM")
    print("=" * 52)
    print(f"  Stream URL  : {config.STREAM_URL}")
    print(f"  Confidence  : {config.CONFIDENCE_THRESHOLD}")
    print(f"  Frame Skip  : Every {config.FRAME_SKIP} frames")
    print(f"  Alert Pause : {config.ALERT_COOLDOWN_SECONDS}s cooldown")
    print("─" * 52)
    print("  Controls: Q=Quit | S=Screenshot | R=Reset | SPACE=Pause")
    print("=" * 52)

    stream   = StreamReader(config.STREAM_URL)
    detector = SurvivorDetector(
        model_path        = config.MODEL_PATH,
        confidence_thresh = config.CONFIDENCE_THRESHOLD,
        frame_skip        = config.FRAME_SKIP,
        screenshot_dir    = config.SCREENSHOT_DIR,
        input_width       = config.INPUT_WIDTH,
        input_height      = config.INPUT_HEIGHT,
    )
    logger      = SessionLogger(log_dir="logs")
    fps_tracker = FPSTracker(window=30)

    paused           = False
    frame_number     = 0
    total_detections = 0

    if not stream.start():
        print("[Main] ❌ Cannot start stream. Check IP Webcam app and URL.")
        sys.exit(1)

    print("\n[Main] 🚁 Drone Rescue System is LIVE. Monitoring for survivors...\n")

    try:
        while stream.is_running():
            frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_number += 1
            fps_tracker.tick()

            if paused:
                display = draw_paused(frame.copy())
                cv2.imshow("AI Rescue Drone — Survivor Monitor", display)
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):
                    paused = False
                    print("[Main] ▶  Resumed.")
                elif key == ord('q'):
                    break
                continue

            result       = detector.process_frame(frame)
            display_frame = draw_fps(result.annotated_frame, fps_tracker.get_fps())
            cv2.imshow("AI Rescue Drone — Survivor Monitor", display_frame)

            if result.was_yolo_run and result.survivor_count > 0:
                total_detections += result.survivor_count
                logger.log(
                    timestamp       = result.timestamp,
                    frame_num       = frame_number,
                    survivor_count  = result.survivor_count,
                    confidences     = result.confidences,
                    screenshot_path = result.screenshot_path,
                )
                print(
                    f"[Main] 🔴 ALERT | Frame {frame_number:05d} | "
                    f"Survivors: {result.survivor_count} | "
                    f"Conf: {result.confidences} | "
                    f"Time: {result.timestamp}"
                )
                trigger_alert(
                    survivor_count  = result.survivor_count,
                    screenshot_path = result.screenshot_path,
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[Main] Q pressed — shutting down...")
                break
            elif key == ord('s'):
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(config.SCREENSHOT_DIR, f"manual_{ts}.jpg")
                cv2.imwrite(path, display_frame)
                print(f"[Main] 📸 Manual screenshot: {path}")
            elif key == ord('r'):
                detector.reset_counter()
                frame_number     = 0
                total_detections = 0
                print("[Main] 🔄 Session counters reset.")
            elif key == ord(' '):
                paused = True
                print("[Main] ⏸  Paused.")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user (Ctrl+C).")
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        stats = detector.get_stats()
        print("\n" + "=" * 52)
        print("          SESSION SUMMARY")
        print("=" * 52)
        print(f"  Frames Processed  : {stats['frames_processed']}")
        print(f"  Screenshots Saved : {stats['screenshots_saved']}")
        print(f"  Total Detections  : {total_detections}")
        print(f"  Log File          : logs/")
        print("=" * 52)


if __name__ == "__main__":
    main()
