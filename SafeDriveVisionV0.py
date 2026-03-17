import argparse
from collections import deque
from functools import lru_cache
import importlib.util
import math
import shutil
import sys
import threading
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import distance as dist

try:
    import dlib
except ImportError as exc:
    raise SystemExit(
        "dlib kurulu değil. Scripti çalıştırmadan önce sanal ortama kurun."
    ) from exc

try:
    import pygame
except ImportError:
    pygame = None

try:
    import torch
except ImportError:
    torch = None

try:
    mediapipe_spec = importlib.util.find_spec("mediapipe")
    if mediapipe_spec is not None and mediapipe_spec.origin is not None:
        mediapipe_dir = Path(mediapipe_spec.origin).resolve().parent
        if not str(mediapipe_dir).isascii():
            mediapipe_ascii_root = Path(tempfile.gettempdir()) / "safedrivevision_sitepackages"
            mediapipe_ascii_dir = mediapipe_ascii_root / "mediapipe"
            if not mediapipe_ascii_dir.exists():
                shutil.copytree(mediapipe_dir, mediapipe_ascii_dir, dirs_exist_ok=True)
            if str(mediapipe_ascii_root) not in sys.path:
                sys.path.insert(0, str(mediapipe_ascii_root))
    import mediapipe as mp
except ImportError:
    mp = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


PROJECT_ROOT = Path(__file__).resolve().parent
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
)

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (-30.0, -125.0, -30.0),
        (30.0, -125.0, -30.0),
        (-60.0, -70.0, -60.0),
        (60.0, -70.0, -60.0),
        (0.0, -330.0, -65.0),
    ],
    dtype="double",
)

LOOK_AHEAD_MIN_ANGLE = 75
LOOK_AHEAD_MAX_ANGLE = 110
DEFAULT_EYE_AR_THRESHOLD = 0.26
MOUTH_AR_THRESHOLD = 0.7
EYE_SMOOTHING_FRAMES = 5
EYE_WARNING_FRAMES = 3
EYE_SOUND_FRAMES = 6
INVALID_FRAME_MEAN_MAX = 5.0
INVALID_FRAME_STD_MAX = 10.0
CAMERA_WARMUP_FRAMES = 15
CAMERA_OPEN_ATTEMPTS = 4
MAX_INVALID_FRAMES_BEFORE_REOPEN = 30
SMOKING_WARNING_FRAMES = 5
SMOKING_ALERT_FRAMES = 10
HAND_TO_MOUTH_RATIO = 0.45
CIGARETTE_MIN_LENGTH_RATIO = 0.18
CIGARETTE_MAX_THICKNESS_RATIO = 0.12
CIGARETTE_MIN_ASPECT_RATIO = 2.5
FRAME_COLOR = (0, 0, 0)
NOSE_GUIDE_COLOR = (128, 128, 128)
TEXT_COLOR = (0, 0, 0)
LANDMARK_COLOR = (128, 128, 128)
CONTOUR_COLOR = (0, 0, 0)


class SoundPlayer:
    def __init__(self, enabled=True):
        self.enabled = enabled and pygame is not None
        self.sounds = {
            "eye": (PROJECT_ROOT / "eye_tr.mp3", 10),
            "regarder": (PROJECT_ROOT / "look_tr.mp3", 10),
            "reposer": (PROJECT_ROOT / "rest_tr.mp3", 15),
            "phone": (PROJECT_ROOT / "phone_tr.mp3", 15),
            "smoking": (PROJECT_ROOT / "smoking_tr.mp3", 15),
            "welcome": (PROJECT_ROOT / "welcome_tr.mp3", 0),
        }
        self.last_played = {key: 0.0 for key in self.sounds}

        if self.enabled:
            try:
                pygame.mixer.init()
            except Exception:
                self.enabled = False

    def play(self, sound_key):
        if not self.enabled:
            return

        sound_file, delay = self.sounds[sound_key]
        if not sound_file.exists():
            return

        now = time.time()
        if now - self.last_played[sound_key] <= delay:
            return

        try:
            pygame.mixer.music.load(str(sound_file))
            pygame.mixer.music.play()
            self.last_played[sound_key] = now
        except Exception:
            self.enabled = False

    def play_async(self, sound_key):
        if not self.enabled:
            return
        thread = threading.Thread(target=self.play, args=(sound_key,), daemon=True)
        thread.start()


def parse_args():
    parser = argparse.ArgumentParser(description="Temel SafeDriveVision demosunu çalıştır.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-backend", choices=["auto", "dshow", "msmf"], default="auto")
    parser.add_argument(
        "--landmark-model",
        type=Path,
        default=PROJECT_ROOT / "shape_predictor_81_face_landmarks (1).dat",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "yolov5m.pt",
    )
    parser.add_argument("--disable-yolo", action="store_true")
    parser.add_argument("--disable-sound", action="store_true")
    parser.add_argument("--disable-smoking-detection", action="store_true")
    parser.add_argument("--eye-threshold", type=float, default=DEFAULT_EYE_AR_THRESHOLD)
    return parser.parse_args()


def get_camera_matrix(size):
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    return np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )


def is_rotation_matrix(rotation):
    rotation_t = np.transpose(rotation)
    should_be_identity = np.dot(rotation_t, rotation)
    identity = np.identity(3, dtype=rotation.dtype)
    return np.linalg.norm(identity - should_be_identity) < 1e-6


def rotation_matrix_to_euler_angles(rotation):
    if not is_rotation_matrix(rotation):
        raise ValueError("Invalid rotation matrix")

    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-6
    if not singular:
        x_value = math.atan2(rotation[2, 1], rotation[2, 2])
        y_value = math.atan2(-rotation[2, 0], sy)
        z_value = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        x_value = math.atan2(-rotation[1, 2], rotation[1, 1])
        y_value = math.atan2(-rotation[2, 0], sy)
        z_value = 0
    return np.array([x_value, y_value, z_value])


def get_head_tilt_and_coords(size, image_points, frame_height):
    camera_matrix = get_camera_matrix(size)
    dist_coeffs = np.zeros((4, 1))
    _, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    nose_end_point_2d, _ = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]),
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs,
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    head_tilt_degree = abs(
        -180 - np.rad2deg(rotation_matrix_to_euler_angles(rotation_matrix)[0])
    )
    starting_point = (int(image_points[0][0]), int(image_points[0][1]))
    ending_point = (
        int(nose_end_point_2d[0][0][0]),
        int(nose_end_point_2d[0][0][1]),
    )
    ending_point_alternate = (ending_point[0], frame_height // 2)
    return head_tilt_degree, starting_point, ending_point, ending_point_alternate


def eye_aspect_ratio(eye):
    a_dist = dist.euclidean(eye[1], eye[5])
    b_dist = dist.euclidean(eye[2], eye[4])
    c_dist = dist.euclidean(eye[0], eye[3])
    return (a_dist + b_dist) / (2.0 * c_dist)


def mouth_aspect_ratio(mouth):
    a_dist = dist.euclidean(mouth[2], mouth[10])
    b_dist = dist.euclidean(mouth[4], mouth[8])
    c_dist = dist.euclidean(mouth[0], mouth[6])
    return (a_dist + b_dist) / (2.0 * c_dist)


def nose_aspect_ratio(nose):
    vertical_distance = dist.euclidean(nose[0], nose[2])
    depth_distance = dist.euclidean(nose[0], nose[1])
    return depth_distance / vertical_distance


def calculate_head_angle(eye_left, eye_right, nose_tip):
    eye_center = (eye_left + eye_right) / 2
    vector_nose = nose_tip - eye_center
    vector_horizontal = eye_right - eye_left
    vector_horizontal[1] = 0
    vector_nose_normalized = vector_nose / np.linalg.norm(vector_nose)
    vector_horizontal_normalized = vector_horizontal / np.linalg.norm(vector_horizontal)
    angle_rad = np.arccos(
        np.clip(np.dot(vector_nose_normalized, vector_horizontal_normalized), -1.0, 1.0)
    )
    return np.degrees(angle_rad)


def to_ascii_fallback(text):
    replacements = str.maketrans(
        {
            "ç": "c",
            "Ç": "C",
            "ğ": "g",
            "Ğ": "G",
            "ı": "i",
            "İ": "I",
            "ö": "o",
            "Ö": "O",
            "ş": "s",
            "Ş": "S",
            "ü": "u",
            "Ü": "U",
        }
    )
    return text.translate(replacements)


@lru_cache(maxsize=8)
def get_font(font_size):
    if ImageFont is None:
        return None
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), font_size)
    return ImageFont.load_default()


def draw_texts(frame, text_items):
    if not text_items:
        return frame

    if Image is None or ImageDraw is None or ImageFont is None:
        for text, position, color, font_size in text_items:
            scale = max(font_size / 32.0, 0.5)
            cv2.putText(
                frame,
                to_ascii_fallback(text),
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                2,
            )
        return frame

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for text, position, color, font_size in text_items:
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=get_font(font_size), fill=rgb_color)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def create_hand_detector(disable_smoking_detection):
    if disable_smoking_detection or mp is None:
        return None
    try:
        return mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as exc:
        print(f"[UYARI] Sigara heuristigi icin el modeli yuklenemedi: {exc}")
        return None


def detect_smoking_like_behavior(hand_results, frame_shape, mouth_points, face_width):
    if hand_results is None or hand_results.multi_hand_landmarks is None:
        return False

    frame_height, frame_width = frame_shape[:2]
    mouth_center = mouth_points.mean(axis=0)
    mouth_radius = max(face_width * HAND_TO_MOUTH_RATIO, 50.0)
    fingertip_ids = (4, 8, 12, 16, 20)

    for hand_landmarks in hand_results.multi_hand_landmarks:
        fingertip_points = []
        for landmark_id in fingertip_ids:
            landmark = hand_landmarks.landmark[landmark_id]
            fingertip_points.append(
                np.array([landmark.x * frame_width, landmark.y * frame_height], dtype=np.float32)
            )

        near_tip_count = sum(
            np.linalg.norm(point - mouth_center) <= mouth_radius for point in fingertip_points
        )
        if near_tip_count >= 2:
            return True

    return False


def detect_cigarette_shape_near_mouth(frame, mouth_points, face_width):
    frame_height, frame_width = frame.shape[:2]
    mouth_center = mouth_points.mean(axis=0)

    pad_x = int(max(face_width * 0.45, 40))
    pad_y = int(max(face_width * 0.25, 25))
    x1 = max(int(mouth_center[0] - pad_x), 0)
    y1 = max(int(mouth_center[1] - pad_y), 0)
    x2 = min(int(mouth_center[0] + pad_x), frame_width)
    y2 = min(int(mouth_center[1] + pad_y), frame_height)

    if x2 <= x1 or y2 <= y1:
        return False

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_length = face_width * CIGARETTE_MIN_LENGTH_RATIO
    max_thickness = max(face_width * CIGARETTE_MAX_THICKNESS_RATIO, 8.0)

    for contour in contours:
        x_pos, y_pos, width, height = cv2.boundingRect(contour)
        long_side = max(width, height)
        short_side = max(1, min(width, height))
        aspect_ratio = long_side / short_side

        if long_side < min_length:
            continue
        if short_side > max_thickness:
            continue
        if aspect_ratio < CIGARETTE_MIN_ASPECT_RATIO:
            continue

        contour_center = np.array(
            [x1 + x_pos + width / 2.0, y1 + y_pos + height / 2.0], dtype=np.float32
        )
        if np.linalg.norm(contour_center - mouth_center) <= face_width * 0.6:
            return True

    return False


def load_phone_model(weights_path, disable_yolo):
    if disable_yolo:
        return None
    if torch is None:
        print("[UYARI] torch kurulu değil, telefon tespiti kapatıldı.")
        return None
    if not weights_path.exists():
        print(f"[UYARI] YOLO ağırlık dosyası bulunamadı: {weights_path}. Telefon tespiti kapatıldı.")
        return None

    try:
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(weights_path),
            force_reload=False,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model
    except Exception as exc:
        print(f"[UYARI] YOLO modeli yüklenemedi: {exc}")
        return None


def validate_inputs(args):
    if not args.landmark_model.exists():
        raise FileNotFoundError(
            f"Landmark modeli bulunamadı: {args.landmark_model}"
        )


def prepare_landmark_model(landmark_model):
    cache_dir = Path(tempfile.gettempdir()) / "safedrivevision"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_model = cache_dir / "shape_predictor_81_face_landmarks.dat"
    if (not cached_model.exists()) or landmark_model.stat().st_mtime > cached_model.stat().st_mtime:
        shutil.copy2(landmark_model, cached_model)
    return cached_model


def open_camera(camera_index, backend_name):
    backends = []
    fallback_cap = None
    if backend_name == "dshow":
        backends.append(cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else None)
    elif backend_name == "msmf":
        backends.append(cv2.CAP_MSMF if hasattr(cv2, "CAP_MSMF") else None)
    else:
        backends.append(None)
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)

    for backend in backends:
        for _attempt in range(CAMERA_OPEN_ATTEMPTS):
            if backend_name in ("dshow", "msmf") and backend is None:
                continue
            if backend is None:
                cap = cv2.VideoCapture(camera_index)
            else:
                cap = cv2.VideoCapture(camera_index, backend)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if not cap.isOpened():
                cap.release()
                continue

            if fallback_cap is None:
                fallback_cap = cap

            valid_frame = None
            for _ in range(CAMERA_WARMUP_FRAMES):
                ret, frame = cap.read()
                if ret and not is_invalid_frame(frame):
                    valid_frame = frame
                    break

            if valid_frame is not None or (
                backend_name in ("dshow", "msmf") and _attempt == CAMERA_OPEN_ATTEMPTS - 1
            ):
                return cap
            if cap is not fallback_cap:
                cap.release()
            time.sleep(0.5)

    if fallback_cap is not None:
        return fallback_cap

    raise RuntimeError(f"Kamera {camera_index} açılamadı.")


def is_invalid_frame(frame):
    if frame is None or frame.size == 0:
        return True
    return float(frame.mean()) <= INVALID_FRAME_MEAN_MAX and float(frame.std()) <= INVALID_FRAME_STD_MAX


def main():
    args = parse_args()
    validate_inputs(args)

    sound_player = SoundPlayer(enabled=not args.disable_sound)
    detector = dlib.get_frontal_face_detector()
    predictor_model = prepare_landmark_model(args.landmark_model)
    predictor = dlib.shape_predictor(str(predictor_model))
    phone_model = load_phone_model(args.yolo_weights, args.disable_yolo)
    hand_detector = create_hand_detector(args.disable_smoking_detection)

    cap = open_camera(args.camera_index, args.camera_backend)

    eye_counter = 0
    phone_counter = 0
    look_counter = 0
    repeat_counter = 0
    smoking_counter = 0
    invalid_frame_counter = 0
    ear_history = deque(maxlen=EYE_SMOOTHING_FRAMES)

    sound_player.play_async("welcome")

    while True:
        ret, frame = cap.read()
        if not ret or is_invalid_frame(frame):
            invalid_frame_counter += 1
            fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            error_texts = [
                ("Kamera goruntusu bozuk veya alinamiyor", (20, 40), TEXT_COLOR, 26),
                (f"Kamera indeksi: {args.camera_index}", (20, 90), TEXT_COLOR, 22),
                (f"Backend: {args.camera_backend}", (20, 120), TEXT_COLOR, 22),
                ("Farkli backend dene: --camera-backend dshow/msmf", (20, 170), TEXT_COLOR, 20),
                ("Windows Kamera uygulamasinda da kontrol et", (20, 200), TEXT_COLOR, 20),
            ]
            if invalid_frame_counter >= MAX_INVALID_FRAMES_BEFORE_REOPEN:
                cap.release()
                time.sleep(1)
                cap = open_camera(args.camera_index, args.camera_backend)
                invalid_frame_counter = 0
            if invalid_frame_counter >= 3:
                fallback_frame = draw_texts(fallback_frame, error_texts)
                cv2.imshow("SafeDriveVision", fallback_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            if not ret:
                continue
        else:
            invalid_frame_counter = 0

        text_overlays = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hand_detector.process(frame_rgb) if hand_detector is not None else None
        faces = detector(gray, 0)

        if len(faces) == 0:
            smoking_counter = 0
            text_overlays.append(("İleri bak", (10, 30), TEXT_COLOR, 28))
            sound_player.play_async("regarder")

        if phone_model is not None:
            results = phone_model(frame)
            detections = results.xyxy[0]
            for detection in detections:
                if int(detection[5]) != 67:
                    continue
                x1, y1, x2, y2 = map(int, detection[:4])
                conf = float(detection[4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), FRAME_COLOR, 2)
                text_overlays.append((f"Cep telefonu {conf:.2f}", (x1, y1 - 24), TEXT_COLOR, 22))
                phone_counter += 1
                if phone_counter >= 3:
                    text_overlays.append(("Telefonu bırak", (x1, y1 - 50), TEXT_COLOR, 24))
                    sound_player.play_async("phone")
                    phone_counter = 0

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks_points = np.array([(point.x, point.y) for point in landmarks.parts()])

            x_pos, y_pos, width, height = (
                face.left(),
                face.top(),
                face.width(),
                face.height(),
            )
            cv2.rectangle(frame, (x_pos, y_pos), (x_pos + width, y_pos + height), FRAME_COLOR, 2)

            image_points = np.array(
                [
                    landmarks_points[30],
                    landmarks_points[36],
                    landmarks_points[45],
                    landmarks_points[48],
                    landmarks_points[54],
                    landmarks_points[8],
                ],
                dtype="double",
            )

            for point in landmarks_points:
                cv2.circle(frame, (point[0], point[1]), 2, LANDMARK_COLOR, -1)

            left_eye = landmarks_points[36:42]
            right_eye = landmarks_points[42:48]
            cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, CONTOUR_COLOR, 1)
            cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, CONTOUR_COLOR, 1)
            ear_raw = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            ear_history.append(ear_raw)
            ear = float(np.mean(ear_history))

            mouth = landmarks_points[48:68]
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, CONTOUR_COLOR, 1)
            mar = mouth_aspect_ratio(mouth)

            nose_points = [landmarks_points[27], landmarks_points[30], landmarks_points[33]]
            nar = nose_aspect_ratio(nose_points)
            head_angle = calculate_head_angle(
                np.array(landmarks_points[36]),
                np.array(landmarks_points[45]),
                np.array(landmarks_points[33]),
            )

            frame_height = frame.shape[0]
            head_tilt_degree, start_point, end_point, end_point_alt = get_head_tilt_and_coords(
                frame.shape,
                image_points,
                frame_height,
            )

            text_overlays.append((f"Göz oranı: {ear:.2f}", (10, 20), TEXT_COLOR, 22))
            text_overlays.append((f"Ağız oranı: {mar:.2f}", (10, 45), TEXT_COLOR, 22))
            text_overlays.append((f"Baş açısı: {head_angle:.2f}", (10, 70), TEXT_COLOR, 22))
            text_overlays.append((f"Burun oranı: {nar:.2f}", (10, 95), TEXT_COLOR, 22))
            text_overlays.append((f"Baş eğimi: {head_tilt_degree:.2f} derece", (10, 120), TEXT_COLOR, 22))
            cv2.line(frame, start_point, end_point, NOSE_GUIDE_COLOR, 2)
            cv2.line(frame, start_point, end_point_alt, NOSE_GUIDE_COLOR, 2)

            face_width = max(1, width)
            hand_near_mouth = detect_smoking_like_behavior(hand_results, frame.shape, mouth, face_width)
            object_near_mouth = detect_cigarette_shape_near_mouth(frame, mouth, face_width)
            if hand_near_mouth and object_near_mouth:
                smoking_counter += 2
            elif hand_near_mouth:
                smoking_counter += 1
            else:
                smoking_counter = max(smoking_counter - 1, 0)

            if smoking_counter >= SMOKING_WARNING_FRAMES:
                text_overlays.append(("Sigara icme supheli", (x_pos, y_pos - 105), TEXT_COLOR, 28))
            if smoking_counter >= SMOKING_ALERT_FRAMES:
                sound_player.play_async("smoking")
                smoking_counter = SMOKING_WARNING_FRAMES

            if head_angle < LOOK_AHEAD_MIN_ANGLE or head_angle > LOOK_AHEAD_MAX_ANGLE:
                text_overlays.append(("İleri bak", (x_pos, y_pos - 35), TEXT_COLOR, 30))
                look_counter += 1
                if look_counter >= 6:
                    sound_player.play_async("regarder")
                    look_counter = 0
            else:
                look_counter = 0

            if ear < args.eye_threshold:
                eye_counter += 1
                if eye_counter >= EYE_WARNING_FRAMES:
                    text_overlays.append(("Gözler kapalı!", (x_pos, y_pos - 35), TEXT_COLOR, 30))
                if eye_counter >= EYE_SOUND_FRAMES:
                    sound_player.play_async("eye")
                    repeat_counter += 1
                    eye_counter = 0
                    if repeat_counter >= 3:
                        sound_player.play_async("reposer")
                        repeat_counter = 0
            else:
                eye_counter = 0
                repeat_counter = 0

            if mar > MOUTH_AR_THRESHOLD:
                sound_player.play_async("reposer")
                text_overlays.append(("Esneme!", (x_pos, y_pos - 70), TEXT_COLOR, 30))

        frame = draw_texts(frame, text_overlays)
        cv2.imshow("SafeDriveVision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
