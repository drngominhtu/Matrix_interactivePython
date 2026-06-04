from __future__ import annotations

# Ensure script runs from its own directory and prefer the project's venv Python
import os
import sys

# change CWD to script directory so `python main.py` works from any folder
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
try:
	if os.getcwd() != _SCRIPT_DIR:
		os.chdir(_SCRIPT_DIR)
except Exception:
	pass

# If a local virtualenv exists, re-exec using its Python interpreter so
# you don't need to manually Activate/Source the venv before running.
_possible_venv = [
	os.path.join(_SCRIPT_DIR, ".venv", "Scripts", "python.exe"),
	os.path.join(_SCRIPT_DIR, ".venv", "bin", "python"),
]
for _venv_py in _possible_venv:
	try:
		_venv_abs = os.path.abspath(_venv_py)
		if os.path.exists(_venv_abs) and os.path.abspath(sys.executable) != _venv_abs:
			os.execv(_venv_abs, [_venv_abs] + sys.argv)
	except Exception:
		# If re-exec fails, fall back to current interpreter
		pass

import argparse
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


ASCII_TABLE = np.asarray(list("@%#*+=-:. "))


@dataclass
class Point:
	x: int
	y: int


def order_points_clockwise(points: list[Point]) -> np.ndarray:
	ngominhtu = np.array([[p.x, p.y] for p in points], dtype=np.float32)
	center = ngominhtu.mean(axis=0)
	angles = np.arctan2(ngominhtu[:, 1] - center[1], ngominhtu[:, 0] - center[0])
	ordered = ngominhtu[np.argsort(angles)]
	return ordered.astype(np.int32)


class AsciiLayerRenderer:
	def __init__(self, columns: int = 110) -> None:
		self.columns = max(40, columns)
		self.font = cv2.FONT_HERSHEY_PLAIN

	def render(self, frame_bgr: np.ndarray) -> np.ndarray:
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
		height, width = gray.shape

		rows = max(24, int(self.columns * height / max(width, 1) * 0.68))
		cell_w = width / self.columns
		cell_h = height / rows

		small = cv2.resize(gray, (self.columns, rows), interpolation=cv2.INTER_AREA)
		indices = np.clip((small.astype(np.float32) / 256.0 * len(ASCII_TABLE)).astype(np.int32), 0, len(ASCII_TABLE) - 1)

		canvas = np.zeros_like(frame_bgr)
		font_scale = max(0.35, min(cell_w / 14.0, cell_h / 22.0))
		thickness = 1

		for row in range(rows):
			y = int((row + 1) * cell_h)
			for col in range(self.columns):
				x = int(col * cell_w)
				char = ASCII_TABLE[indices[row, col]]
				cv2.putText(canvas, char, (x, y), self.font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)

		return canvas


class HandQuadDetector:
	def __init__(self) -> None:
		self.hands = mp.solutions.hands.Hands(
			static_image_mode=False,
			max_num_hands=2,
			min_detection_confidence=0.65,
			min_tracking_confidence=0.65,
		)
		self.drawer = mp.solutions.drawing_utils
		self.styles = mp.solutions.drawing_styles

	def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, list[Point], int]:
		rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		results = self.hands.process(rgb)
		hand_points: list[tuple[str, Point, Point]] = []

		if not results.multi_hand_landmarks:
			return frame_bgr, [], 0

		h, w = frame_bgr.shape[:2]
		handedness_list = results.multi_handedness or []

		for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
			label = "Unknown"
			if idx < len(handedness_list):
				label = handedness_list[idx].classification[0].label

			thumb = hand_landmarks.landmark[4]
			index = hand_landmarks.landmark[8]
			thumb_point = Point(int(thumb.x * w), int(thumb.y * h))
			index_point = Point(int(index.x * w), int(index.y * h))
			hand_points.append((label, thumb_point, index_point))

			self.drawer.draw_landmarks(
				frame_bgr,
				hand_landmarks,
				mp.solutions.hands.HAND_CONNECTIONS,
				self.styles.get_default_hand_landmarks_style(),
				self.styles.get_default_hand_connections_style(),
			)

		left_thumb = left_index = right_thumb = right_index = None
		if len(hand_points) >= 2:
			hand_points.sort(key=lambda item: item[1].x)
			left_thumb, left_index = hand_points[0][1], hand_points[0][2]
			right_thumb, right_index = hand_points[-1][1], hand_points[-1][2]
		elif len(hand_points) == 1:
			label, thumb_point, index_point = hand_points[0]
			if label == "Left":
				left_thumb, left_index = thumb_point, index_point
			elif label == "Right":
				right_thumb, right_index = thumb_point, index_point

		quad_points = [point for point in [left_thumb, left_index, right_index, right_thumb] if point is not None]
		if len(quad_points) == 4:
			return frame_bgr, quad_points, len(results.multi_hand_landmarks)

		return frame_bgr, [], len(results.multi_hand_landmarks)


class CameraAsciiQuadApp:
	def __init__(self, camera_index: int, width: int, height: int, ascii_columns: int, mirror: bool) -> None:
		self.camera_index = camera_index
		self.width = width
		self.height = height
		self.mirror = mirror
		print(f"[INFO] Opening camera {camera_index}...")
		self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
		if not self.capture.isOpened():
			print(f"[WARN] CAP_DSHOW failed, trying default...")
			self.capture = cv2.VideoCapture(camera_index)
		
		if not self.capture.isOpened():
			raise RuntimeError(f"Cannot open camera {camera_index}. Check if camera is available.")
		
		print(f"[INFO] Camera opened, setting properties...")
		self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		self.capture.set(cv2.CAP_PROP_FPS, 30)
		
		print(f"[INFO] Initializing hand detector...")
		self.ascii_renderer = AsciiLayerRenderer(columns=ascii_columns)
		self.hand_detector = HandQuadDetector()
		self.last_time = time.time()
		self.fps = 0.0
		print(f"[INFO] CameraAsciiQuadApp initialized successfully")

	def _blend_layers(self, raw: np.ndarray, ascii_layer: np.ndarray, quad: list[Point]) -> np.ndarray:
		base = raw.copy()

		if len(quad) == 4:
			polygon = order_points_clockwise(quad)
			mask = np.zeros(raw.shape[:2], dtype=np.uint8)
			cv2.fillConvexPoly(mask, polygon, 255)
			mask = cv2.GaussianBlur(mask, (21, 21), 0)

			alpha = (mask.astype(np.float32) / 255.0)[..., None]
			gray_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
			gray_raw = cv2.cvtColor(gray_raw, cv2.COLOR_GRAY2BGR)
			ascii_gray = cv2.cvtColor(ascii_layer, cv2.COLOR_BGR2GRAY)
			ascii_gray = cv2.cvtColor(ascii_gray, cv2.COLOR_GRAY2BGR)
			ascii_only = cv2.addWeighted(gray_raw, 0.4, ascii_gray, 0.6, 0.0)
			base = (raw.astype(np.float32) * (1.0 - alpha) + ascii_only.astype(np.float32) * alpha).astype(np.uint8)

			for point in quad:
				cv2.circle(base, (point.x, point.y), 6, (255, 255, 255), -1, cv2.LINE_AA)

		return base

	def run(self) -> None:
		if not self.capture.isOpened():
			raise RuntimeError(f"Cannot open camera {self.camera_index}")

		try:
			print(f"[INFO] Creating window...")
			cv2.namedWindow("Camera ASCII Quad", cv2.WINDOW_NORMAL)
			cv2.resizeWindow("Camera ASCII Quad", self.width, self.height)
			print(f"[INFO] Window created, starting frame loop...")

			while True:
				ok, frame = self.capture.read()
				if not ok:
					print(f"[WARN] Failed to read frame from camera")
					break

				if self.mirror:
					frame = cv2.flip(frame, 1)

				raw = frame.copy()
				ascii_layer = self.ascii_renderer.render(raw)
				_, quad_points, hand_count = self.hand_detector.detect(frame.copy())
				output = self._blend_layers(raw, ascii_layer, quad_points)

				now = time.time()
				delta = now - self.last_time
				self.last_time = now
				if delta > 0:
					self.fps = 0.9 * self.fps + 0.1 * (1.0 / delta) if self.fps else 1.0 / delta

				self._draw_hud(output, hand_count, len(quad_points), self.fps)
				cv2.imshow("Camera ASCII Quad", output)

				key = cv2.waitKey(1) & 0xFF
				if key in (27, ord("q")):
					break
		finally:
			print(f"[INFO] Closing resources...")
			self.capture.release()
			cv2.destroyAllWindows()
			print(f"[INFO] Resources closed")

	def _draw_hud(self, frame: np.ndarray, hand_count: int, quad_size: int, fps: float) -> None:
		status = f"hands: {hand_count}  quad: {'on' if quad_size == 4 else 'off'}  fps: {fps:.1f}"
		cv2.rectangle(frame, (12, 12), (520, 72), (0, 0, 0), -1)
		cv2.putText(frame, status, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(frame, "ESC/Q to quit", (24, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Camera ASCII overlay with hand quad interaction")
	parser.add_argument("--camera", type=int, default=0, help="Camera index")
	parser.add_argument("--width", type=int, default=1280, help="Capture width")
	parser.add_argument("--height", type=int, default=720, help="Capture height")
	parser.add_argument("--ascii-columns", type=int, default=160, help="ASCII columns")
	parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal mirroring")
	return parser.parse_args()


def main() -> None:
	try:
		args = parse_args()
		print(f"[INFO] Initializing app with camera {args.camera}")
		app = CameraAsciiQuadApp(
			camera_index=args.camera,
			width=args.width,
			height=args.height,
			ascii_columns=args.ascii_columns,
			mirror=not args.no_mirror,
		)
		print("[INFO] App initialized successfully, starting...")
		app.run()
	except Exception as e:
		print(f"[ERROR] Application failed: {e}", flush=True)
		import traceback
		traceback.print_exc()
		input("Press Enter to exit...")


if __name__ == "__main__":
	main()
