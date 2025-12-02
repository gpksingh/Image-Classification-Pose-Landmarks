import os
import cv2
import numpy as np

# lightweight defaults; heavy libs imported in _lazy_init
pose_labels = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]

def _lazy_init(model_h5="yoga_pose_model.h5", model_onnx="yoga_pose_model.onnx"):
    """Lazy-load mediapipe and a model (Keras h5 preferred, fallback to ONNX)."""
    if getattr(_lazy_init, "initialized", False):
        return

    # mediapipe
    try:
        import mediapipe as mp  # type: ignore
        _lazy_init.mp = mp
    except Exception:
        _lazy_init.mp = None

    # try Keras model
    _lazy_init.model_type = None
    _lazy_init.model = None
    if os.path.exists(model_h5):
        try:
            import tensorflow as tf  # type: ignore
            _lazy_init.model = tf.keras.models.load_model(model_h5)
            _lazy_init.model_type = "keras"
        except Exception:
            _lazy_init.model = None
            _lazy_init.model_type = None

    # try ONNX runtime if no keras model
    if _lazy_init.model_type is None and os.path.exists(model_onnx):
        try:
            import onnxruntime as ort  # type: ignore
            sess = ort.InferenceSession(model_onnx, providers=["CPUExecutionProvider"])
            _lazy_init.ort_session = sess
            _lazy_init.ort_input = sess.get_inputs()[0].name
            _lazy_init.ort_output = sess.get_outputs()[0].name
            _lazy_init.model_type = "onnx"
        except Exception:
            _lazy_init.ort_session = None
            _lazy_init.model_type = None

    # create MediaPipe pose (must run in main thread)
    if getattr(_lazy_init, "mp", None) is not None:
        mp = _lazy_init.mp
        try:
            _lazy_init.pose = mp.solutions.pose.Pose(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            _lazy_init.drawing = mp.solutions.drawing_utils
        except Exception:
            _lazy_init.pose = None
            _lazy_init.drawing = None
    else:
        _lazy_init.pose = None
        _lazy_init.drawing = None

    _lazy_init.initialized = True

def _predict_from_model(input_data):
    """Return 1d numpy scores or None."""
    mt = getattr(_lazy_init, "model_type", None)
    if mt == "keras":
        try:
            pred = _lazy_init.model.predict(input_data, verbose=0)
            arr = np.asarray(pred)
            if arr.ndim == 2:
                return arr[0]
            return arr
        except Exception:
            return None


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def is_full_body_visible(landmarks, visibility_threshold=0.6):
    _lazy_init()
    mp = getattr(_lazy_init, "mp", None)
    if mp is None:
        return False
    #testing on left side only
    essential = [
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        # mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        mp.solutions.pose.PoseLandmark.LEFT_HIP,
        # mp.solutions.pose.PoseLandmark.RIGHT_HIP,
        mp.solutions.pose.PoseLandmark.LEFT_KNEE,
        # mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
        mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
        # mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
    ]
    for lm in essential:
        l = landmarks[lm.value]
        if l.visibility < visibility_threshold or not (0 <= l.x <= 1) or not (0 <= l.y <= 1):
            return False
    return True

def get_downdog_feedback(landmarks, w, h):
    _lazy_init()
    mp = getattr(_lazy_init, "mp", None)
    feedback = []
    if mp is None:
        return feedback
    try:
        l_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        l_elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                   landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        l_wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x * w,
                   landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y * h]
        
        left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        if left_elbow_angle < 160:
            feedback.append("Straighten your arms")

        l_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * w,
                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * h]
        l_knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * w,
                  landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h]
        l_ankle = [landmarks[mp.solutions.pose.PoseLandmark.ANKLE.value].x * w,
                  landmarks[mp.solutions.pose.PoseLandmark.ANKLE.value].y * h]
      
        left_leg_angle = calculate_angle(l_hip, l_knee, l_ankle)

        if left_leg_angle < 160:
            feedback.append("Straighten your legs")



        hip_angle = calculate_angle(l_knee, l_hip, l_shoulder)
        if hip_angle < 70:
            feedback.append("Hip angle too low")
        elif hip_angle > 75:
            feedback.append("Hipe angle too high")
    except Exception:
        pass
    if not feedback:
        feedback.append("Good form!")
    return feedback

def get_plank_feedback(landmarks, w, h):
    _lazy_init()
    mp = getattr(_lazy_init, "mp", None)
    feedback = []
    if mp is None:
        return feedback
    try:
        l_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        l_elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                   landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        l_wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x * w,
                   landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y * h]

        r_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        r_elbow = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                   landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        r_wrist = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                   landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y * h]

        left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        if left_elbow_angle < 160 or right_elbow_angle < 160:
            feedback.append("Straighten your arms")

        l_hip = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * w,
                 landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * h]
        r_hip = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w,
                 landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * h]
        l_knee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * w,
                  landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h]
        r_knee = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                  landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h]

        mid_shoulder = np.mean([l_shoulder, r_shoulder], axis=0)
        mid_hip = np.mean([l_hip, r_hip], axis=0)
        mid_knee = np.mean([l_knee, r_knee], axis=0)

        hip_angle = calculate_angle(mid_shoulder, mid_hip, mid_knee)
        if hip_angle < 160:
            feedback.append("Lower your hips")
        elif hip_angle > 190:
            feedback.append("Lift your hips")
    except Exception:
        pass
    if not feedback:
        feedback.append("Good form!")
    return feedback

def process_frame(frame, model_h5="yoga_pose_model.h5", model_onnx="yoga_pose_model.onnx"):
    """
    Process a single BGR frame and return annotated BGR frame.
    Safe to call from app.py (per-frame) or from run_demo() when running file directly.
    """
    _lazy_init(model_h5, model_onnx)
    pose = getattr(_lazy_init, "pose", None)
    mp = getattr(_lazy_init, "mp", None)
    drawing = getattr(_lazy_init, "drawing", None)

    if pose is None:
        return frame

    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    results = pose.process(img_rgb)
    img_rgb.flags.writeable = True
    annotated = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    pose_class = "Not Detected"
    feedback_text = []

    if results and results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        if is_full_body_visible(landmarks):
            keypoints = []
            for lm in landmarks:
                keypoints.extend([lm.x, lm.y, lm.z])
            input_data = np.array(keypoints, dtype=np.float32).reshape(1, -1)
            pred = _predict_from_model(input_data)
            if pred is None:
                pose_class = "Model not available"
            else:
                class_id = int(np.argmax(pred))
                confidence = float(np.max(pred))
                pose_class = pose_labels[class_id] if confidence > 0.7 else "Uncertain"
                if pose_class == "Plank":
                    feedback_text = get_plank_feedback(landmarks, w, h)
                if pose_class == "Downdog":
                    feedback_text = get_downdog_feedback(landmarks, w, h)
        else:
            pose_class = "Full body not visible"

        if drawing is not None and mp is not None:
            drawing.draw_landmarks(annotated, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    else:
        pose_class = "No pose"

    # Draw large centered title + bigger feedback above the webcam image
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title (centered)
    font_scale_title = 1.6
    thickness_title = 3
    title = f"Pose: {pose_class}"
    (title_w, title_h), _ = cv2.getTextSize(title, font, font_scale_title, thickness_title)
    x_title = max(10, (w - title_w) // 2)
    y_title = title_h + 20
    cv2.putText(annotated, title, (x_title, y_title), font, font_scale_title,
                (0, 255, 0), thickness_title, lineType=cv2.LINE_AA)

    # Feedback lines (centered under title)
    font_scale_fb = 1.1
    thickness_fb = 2
    y = y_title + title_h + 10
    for line in feedback_text:
        (lw, lh), _ = cv2.getTextSize(line, font, font_scale_fb, thickness_fb)
        x = max(10, (w - lw) // 2)
        cv2.putText(annotated, line, (x, y), font, font_scale_fb,
                    (0, 0, 255), thickness_fb, lineType=cv2.LINE_AA)
        y += lh + 8

    return annotated

def process_frame_with_info(frame, model_h5="yoga_pose_model.h5", model_onnx="yoga_pose_model.onnx"):
    """Return (annotated_frame, pose_class, feedback_text)."""
    annotated = process_frame(frame, model_h5, model_onnx)
    _lazy_init(model_h5, model_onnx)
    pose = getattr(_lazy_init, "pose", None)
    if pose is None:
        return annotated, "No mediapipe", []
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    results = pose.process(img_rgb)
    img_rgb.flags.writeable = True
    if not (results and results.pose_landmarks):
        return annotated, "No pose", []
    landmarks = results.pose_landmarks.landmark
    if not is_full_body_visible(landmarks):
        return annotated, "Full body not visible", []
    keypoints = []
    for lm in landmarks:
        keypoints.extend([lm.x, lm.y, lm.z])
    pred = _predict_from_model(np.array(keypoints, dtype=np.float32).reshape(1,-1))
    if pred is None:
        return annotated, "Model not available", []
    cid = int(np.argmax(pred)); conf = float(np.max(pred))
    cls = pose_labels[cid] if conf > 0.7 else "Uncertain"
    feedback = get_plank_feedback(landmarks, frame.shape[1], frame.shape[0]) if cls == "Plank" else []
    return annotated, cls, feedback

def close_pose_processor():
    """Release resources."""
    if getattr(_lazy_init, "initialized", False):
        try:
            if getattr(_lazy_init, "pose", None) is not None:
                _lazy_init.pose.close()
        except Exception:
            pass
        for a in ("model", "ort_session", "ort_input", "ort_output", "mp", "pose", "drawing", "model_type", "initialized"):
            if hasattr(_lazy_init, a):
                try:
                    delattr(_lazy_init, a)
                except Exception:
                    pass

def run_demo(camera_index=1):
    """Run single-threaded demo identical to per-frame pipeline used by app.py."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out = process_frame(frame)
            cv2.imshow("Pose Processor (press q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        close_pose_processor()

if __name__ == "__main__":
    run_demo()