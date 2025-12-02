from flask import Flask, render_template, Response
import cv2
from pose_processor import pose_processor
app = Flask(__name__)

# Open webcam (0 = default camera)
camera = cv2.VideoCapture(1)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 👉 If you have mediapipe pose code, apply it to `frame` here
            # frame = process_with_mediapipe(frame)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Stream as multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + pose_processor(frame)
+ b'\r\n')

@app.route('/')
def index():
    # Simple page with video tag
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Route that continuously yields frames
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
