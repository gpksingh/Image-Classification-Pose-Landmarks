# ...existing code...
from flask import Flask, render_template, Response, request
import cv2
import atexit
import numpy as np
import traceback
from pose_processor import process_frame

# ...existing code...
app = Flask(__name__)

# keep existing server-side camera if you still need it
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

# new endpoint: accept single-frame uploads (from browser) and return processed JPEG
@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    try:
        # accept file under 'frame' (FormData from client)
        if 'frame' in request.files:
            file_bytes = request.files['frame'].read()
        else:
            file_bytes = request.get_data()
        if not file_bytes:
            return ('No frame received', 400)

        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return ('Failed to decode image', 400)

        # call your pose processing
        processed = process_frame(img)
        if processed is None:
            processed = img

        ret, jpeg = cv2.imencode('.jpg', processed)
        if not ret:
            return ('Failed to encode', 500)

        return (jpeg.tobytes(), 200, {'Content-Type': 'image/jpeg'})

    except Exception as e:
        traceback.print_exc()
        return (f'Internal error: {e}', 500)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
# ...existing code...