from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from keras.models import load_model
import mediapipe as mp
from scipy.spatial import distance
from lane_finding.main import process_pipeline, calibrate_camera

# ------------------- Flask Setup -------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------- Load Models -------------------
yolo_model = YOLO("models/YOLOv8_Small_RDD.pt")
tsr_model = load_model("models/TSR.h5")

# Traffic Sign Labels
classes = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
    16:'Vehicle > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
    19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
    22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left',
    38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory',
    41:'End of no passing', 42:'End no passing vehicle > 3.5 tons'
}

# ------------------- Home Route -------------------
@app.route('/')
def index():
    return render_template('index.html')

# ------------------- Road Damage Detection -------------------
@app.route('/road', methods=['GET', 'POST'])
def road():
    if request.method == 'POST':
        f = request.files['file']
        if not f:
            return render_template('road.html', uploaded=False, message="No file uploaded!")

        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        # Read and predict
        image = cv2.imread(file_path)
        results = yolo_model(image)

        CLASSES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Potholes"]

        # Draw boxes on image
        for result in results:
            boxes = result.boxes  # no .cpu().numpy()
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(image, f"{CLASSES[cls_id]} {conf:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + filename)
        cv2.imwrite(output_path, image)

        return render_template('road.html', uploaded=True, image_path=output_path)

    return render_template('road.html', uploaded=False)

# ------------------- Traffic Sign Recognition -------------------
@app.route('/traffic', methods=['GET', 'POST'])
def traffic():
    if request.method == 'POST':
        f = request.files['file']
        if not f:
            return render_template('traffic.html', uploaded=False, result="No file selected.")

        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        image = Image.open(file_path).resize((30, 30))
        data = np.expand_dims(np.array(image), axis=0)
        prediction = np.argmax(tsr_model.predict(data))
        result = classes[prediction]

        return render_template('traffic.html', uploaded=True, image_path=file_path, result=result)
    
    return render_template('traffic.html', uploaded=False)

# ------------------- Lane Detection -------------------
# @app.route('/lane', methods=['GET', 'POST'])
# def lane():
#     if request.method == 'POST':
#         f = request.files['file']
#         if not f:
#             return render_template('lane.html', uploaded=False, message="No file uploaded!")

#         filename = secure_filename(f.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         f.save(file_path)

#         # Import your lane detection script dynamically
#         from lane_finding.lane import process_pipeline, calibrate_camera
#         import cv2

#         try:
#             # Camera calibration
#             ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')
#         except Exception as e:
#             return render_template('lane.html', uploaded=False, message=f"Camera calibration failed: {e}")

#         # Process video
#         cap = cv2.VideoCapture(file_path)
#         if not cap.isOpened():
#             return render_template('lane.html', uploaded=False, message="Error opening video file.")

#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'lane_' + filename)
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#         frame_count = 0
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             processed = process_pipeline(frame)
#             out.write(processed)
#             frame_count += 1
#         cap.release()
#         out.release()

#         return render_template('lane.html', uploaded=True, video_path=output_path)

#     return render_template('lane.html', uploaded=False)


# ------------------- Drowsiness Detection (Live in Flask) -------------------
from flask import Response
import mediapipe as mp
from scipy.spatial import distance

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def gen_frames():
    cap = cv2.VideoCapture(0)
    COUNTER = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                leftEye = [(int(face_landmarks.landmark[i].x * w),
                            int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
                rightEye = [(int(face_landmarks.landmark[i].x * w),
                             int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                cv2.polylines(frame, [np.array(leftEye)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [np.array(rightEye)], True, (0, 255, 0), 1)

                if ear < EAR_THRESHOLD:
                    COUNTER += 1
                    if COUNTER >= EAR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    COUNTER = 0

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/drowsiness')
def drowsiness():
    return render_template('drowsiness.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------- Run App -------------------
if __name__ == '__main__':
    app.run(debug=True)
