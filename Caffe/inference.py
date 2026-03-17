import argparse
import ctypes
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOTXT = REPO_ROOT / "Caffe" / "pro.txt"
DEFAULT_MODEL = REPO_ROOT / "Caffe" / "SSD.caffemode"

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
SMILE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

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

LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "cell phone",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SafeDriveVision Caffe + MediaPipe driver monitoring demo"
    )
    parser.add_argument(
        "-p",
        "--prototxt",
        default=str(DEFAULT_PROTOTXT),
        help="Path to the Caffe prototxt file",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the Caffe model weights",
    )
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.7,
        help="Minimum confidence for object detections",
    )
    parser.add_argument(
        "--source",
        default="0",
        help='Camera index, image/video path, or "blank" for a synthetic test frame',
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Resize width for video frames",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames. 0 means run until closed.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without opening OpenCV windows",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default="",
        help="Optional path to save the annotated frame in single-frame runs",
    )
    return parser.parse_args()


def calculate_ear(eye_landmarks: Iterable[Tuple[int, int]]) -> float:
    eye_landmarks = list(eye_landmarks)
    vertical_1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    horizontal = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def rotation_matrix_to_euler_angles(rotation_matrix: np.ndarray) -> np.ndarray:
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
        z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
        z_angle = 0.0
    return np.array([x_angle, y_angle, z_angle], dtype=float)


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    current_height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame
    ratio = width / float(current_width)
    new_height = int(current_height * ratio)
    return cv2.resize(frame, (width, new_height))


def detect_hands_on_wheel(hand_landmarks, mp_hands) -> bool:
    if len(hand_landmarks) != 2:
        return False
    hand1, hand2 = hand_landmarks
    point1 = (
        hand1.landmark[mp_hands.HandLandmark.WRIST].x,
        hand1.landmark[mp_hands.HandLandmark.WRIST].y,
    )
    point2 = (
        hand2.landmark[mp_hands.HandLandmark.WRIST].x,
        hand2.landmark[mp_hands.HandLandmark.WRIST].y,
    )
    return dist.euclidean(point1, point2) < 0.1


def get_hand_side(hand_landmarks, mp_hands) -> str:
    thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
    wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    return "Left Hand" if thumb_tip_x < wrist_x else "Right Hand"


def add_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def analyze_frame(
    frame: np.ndarray,
    net,
    hands,
    face_mesh,
    mp_hands,
    mp_face_mesh,
    mp_drawing,
    confidence_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    frame = frame.copy()
    detected_objects: list[str] = []
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue

        idx = int(detections[0, 0, i, 1])
        if idx < 0 or idx >= len(LABELS):
            continue

        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        start_x, start_y, end_x, end_y = box.astype("int")
        label = f"{LABELS[idx]}: {confidence * 100:.1f}%"
        color = COLORS[idx]

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
        text_y = start_y - 15 if start_y - 15 > 15 else start_y + 15
        cv2.putText(
            frame,
            label,
            (start_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        add_unique(detected_objects, label)
        if LABELS[idx] == "cell phone":
            add_unique(detected_objects, "Cell Phone Detected")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_gray = gray[y : y + h, x : x + w]
        face_color = frame[y : y + h, x : x + w]
        smiles = SMILE_CASCADE.detectMultiScale(
            face_gray, scaleFactor=1.7, minNeighbors=20
        )
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(face_color, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)
            add_unique(detected_objects, "Smile Detected")

    if hand_results.multi_hand_landmarks:
        if detect_hands_on_wheel(hand_results.multi_hand_landmarks, mp_hands):
            add_unique(detected_objects, "Hands on Wheel")
        else:
            add_unique(detected_objects, "Hands off Wheel")

        for hand_landmarks in hand_results.multi_hand_landmarks:
            add_unique(detected_objects, get_hand_side(hand_landmarks, mp_hands))
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            fingertip_distance = dist.euclidean(
                (thumb_tip.x, thumb_tip.y),
                (index_tip.x, index_tip.y),
            )
            if fingertip_distance < 0.05:
                add_unique(detected_objects, "Manipulating Object")
            else:
                add_unique(detected_objects, "No Object Manipulation")

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
            )

            landmarks = face_landmarks.landmark
            left_eye_points = [
                (
                    int(landmarks[idx].x * width),
                    int(landmarks[idx].y * height),
                )
                for idx in LEFT_EYE
            ]
            right_eye_points = [
                (
                    int(landmarks[idx].x * width),
                    int(landmarks[idx].y * height),
                )
                for idx in RIGHT_EYE
            ]
            ear = (calculate_ear(left_eye_points) + calculate_ear(right_eye_points)) / 2.0

            add_unique(detected_objects, "Eyes Closed" if ear < 0.25 else "Eyes Open")

            image_points = np.array(
                [
                    (landmarks[1].x * width, landmarks[1].y * height),
                    (landmarks[33].x * width, landmarks[33].y * height),
                    (landmarks[263].x * width, landmarks[263].y * height),
                    (landmarks[61].x * width, landmarks[61].y * height),
                    (landmarks[291].x * width, landmarks[291].y * height),
                    (landmarks[199].x * width, landmarks[199].y * height),
                ],
                dtype="double",
            )

            focal_length = width
            center = (width / 2.0, height / 2.0)
            camera_matrix = np.array(
                [
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1],
                ],
                dtype="double",
            )
            dist_coeffs = np.zeros((4, 1))
            success, rotation_vector, translation_vector = cv2.solvePnP(
                MODEL_POINTS, image_points, camera_matrix, dist_coeffs
            )

            if success:
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                angles = rotation_matrix_to_euler_angles(rotation_matrix)
                yaw = float(angles[1])
                pitch = float(angles[0])
                roll = float(angles[2])

                if -0.07 <= yaw <= 0.25:
                    add_unique(detected_objects, "Looking Ahead")
                else:
                    add_unique(detected_objects, "Not Looking Ahead")

                if pitch < 1.4:
                    add_unique(detected_objects, "Looking Down")
                elif pitch > 1.6:
                    add_unique(detected_objects, "Looking Up")
                else:
                    add_unique(detected_objects, "Looking Straight")

                add_unique(
                    detected_objects,
                    f"Pitch: {pitch:.2f} Yaw: {yaw:.2f} Roll: {roll:.2f}",
                )

    info_height = max(300, 30 * (len(detected_objects) + 1))
    info_display = np.zeros((info_height, 600, 3), dtype=np.uint8)
    for idx, text in enumerate(detected_objects):
        cv2.putText(
            info_display,
            text,
            (10, 30 * (idx + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return frame, info_display, detected_objects


def resolve_source(source: str):
    if source.lower() == "blank":
        return "blank"
    if source.isdigit():
        return int(source)
    return str(Path(source).expanduser().resolve())


def validate_paths(prototxt_path: Path, model_path: Path) -> None:
    if not prototxt_path.exists():
        raise FileNotFoundError(f"Prototxt file not found: {prototxt_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")


def to_opencv_path(path: Path) -> str:
    path_str = str(path)
    if not path.exists():
        return path_str

    buffer_size = 4096
    buffer = ctypes.create_unicode_buffer(buffer_size)
    result = ctypes.windll.kernel32.GetShortPathNameW(path_str, buffer, buffer_size)
    if result:
        return buffer.value
    return path_str


def write_image(path: str, image: np.ndarray) -> bool:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(output_path.suffix or ".jpg", image)
    if not success:
        return False
    encoded.tofile(str(output_path))
    return True


def main() -> None:
    args = parse_args()
    prototxt_path = Path(args.prototxt).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    validate_paths(prototxt_path, model_path)

    print("[Status] SafeDriveVision Caffe + MediaPipe loading...")
    net = cv2.dnn.readNetFromCaffe(
        to_opencv_path(prototxt_path), to_opencv_path(model_path)
    )
    resolved_source = resolve_source(args.source)

    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(
        min_detection_confidence=0.1, min_tracking_confidence=0.1
    ) as hands, mp_face_mesh.FaceMesh(
        max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:
        if resolved_source == "blank":
            frame = np.zeros((540, 960, 3), dtype=np.uint8)
            annotated, info_display, detected = analyze_frame(
                frame,
                net,
                hands,
                face_mesh,
                mp_hands,
                mp_face_mesh,
                mp_drawing,
                args.confidence,
            )
            print(f"[Status] Smoke test completed. Findings: {detected or ['none']}")
            if args.save_output:
                write_image(args.save_output, annotated)
            if not args.no_display:
                cv2.imshow("Frame", annotated)
                cv2.imshow("Info", info_display)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return

        capture = cv2.VideoCapture(resolved_source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source: {args.source}")

        print("[Status] Video stream started. Press 'q' to quit.")
        processed_frames = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame = resize_frame(frame, args.width)
                annotated, info_display, _ = analyze_frame(
                    frame,
                    net,
                    hands,
                    face_mesh,
                    mp_hands,
                    mp_face_mesh,
                    mp_drawing,
                    args.confidence,
                )
                processed_frames += 1

                if args.save_output and processed_frames == 1:
                    write_image(args.save_output, annotated)

                if not args.no_display:
                    cv2.imshow("Frame", annotated)
                    cv2.imshow("Info", info_display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if args.max_frames and processed_frames >= args.max_frames:
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

        print(f"[Status] Processed {processed_frames} frame(s).")


if __name__ == "__main__":
    main()
