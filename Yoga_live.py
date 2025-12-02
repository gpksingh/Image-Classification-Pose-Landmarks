import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# --- Load your trained model ---
model = tf.keras.models.load_model("yoga_pose_model.h5")

pose_labels = ["Downdog", "Goddess", "Plank", "Tree", "Warrior2"]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


def is_full_body_visible(landmarks, visibility_threshold=0.6):
    """Check if essential body parts are visible and inside frame"""
    essential_points = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
    ]

    for lm in essential_points:
        landmark = landmarks[lm.value]
        if (
            landmark.visibility < visibility_threshold
            or not (0 <= landmark.x <= 1)
            or not (0 <= landmark.y <= 1)
        ):
            return False
    return True


def calculate_angle(a, b, c):
    """Calculate angle between 3 points"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def get_plank_feedback(landmarks, w, h):
    """Check arm straightness and hip alignment in plank"""
    feedback = []

    # --- Shoulder-Elbow-Wrist angles for arms ---
    l_shoulder = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h,
    ]
    l_elbow = [
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h,
    ]
    l_wrist = [
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h,
    ]

    r_shoulder = [
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h,
    ]
    r_elbow = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h,
    ]
    r_wrist = [
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h,
    ]

    left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

    if left_elbow_angle < 160 or right_elbow_angle < 160:
        feedback.append("Straighten your arms")

    # --- Hip alignment: Shoulders-Hips-Knees should be straight ---
    l_hip = [
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h,
    ]
    r_hip = [
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h,
    ]
    l_knee = [
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h,
    ]
    r_knee = [
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h,
    ]

    mid_shoulder = np.mean([l_shoulder, r_shoulder], axis=0)
    mid_hip = np.mean([l_hip, r_hip], axis=0)
    mid_knee = np.mean([l_knee, r_knee], axis=0)

    hip_angle = calculate_angle(mid_shoulder, mid_hip, mid_knee)
    if hip_angle < 160:
        feedback.append("Lower your hips")
    elif hip_angle > 190:
        feedback.append("Lift your hips")

    if not feedback:
        feedback.append("Good form!")

    return feedback


cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    pose_class = "Not Detected"
    feedback_text = []

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        if is_full_body_visible(landmarks):
            keypoints = []
            for lm in landmarks:
                keypoints.extend([lm.x, lm.y, lm.z])

            # Shape: (1, num_features), dtype float32
            input_data = np.array(keypoints, dtype=np.float32).reshape(1, -1)

            # --- ONNX Runtime inference (replaces model.predict) ---
            prediction = session.run(
                [output_name], {input_name: input_data}
            )[0]  # shape (1, num_classes) or (num_classes,)
            if prediction.ndim == 2:
                prediction = prediction[0]

            class_id = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            if confidence > 0.7:
                pose_class = pose_labels[class_id]
            else:
                pose_class = "Uncertain"

            if pose_class == "Plank":
                feedback_text = get_plank_feedback(landmarks, w, h)
        else:
            pose_class = "Full body not visible"

        # Draw landmarks when detected
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
    else:
        pose_class = "No pose detected"

    # Display pose
    cv2.putText(
        image,
        f"Pose: {pose_class}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Display feedback
    y_offset = 60
    for line in feedback_text:
        cv2.putText(
            image,
            line,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        y_offset += 30

    cv2.imshow("Plank Feedback", image)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
