from flask import Flask, render_template, Response
import cv2
import atexit
from pose_processor import process_frame

# ...existing code...
# ...existing code...

app = Flask(__name__)

# Open webcam (0 = default camera)
camera = cv2.VideoCapture(1)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # process the frame with MediaPipe per-frame (non-blocking)
            frame = process_frame(frame)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Stream as multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)