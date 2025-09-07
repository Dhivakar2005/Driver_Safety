import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import winsound   
from playsound import playsound


# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Eye landmark indices (from Mediapipe face mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # left eye landmarks
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # right eye landmarks

# Thresholds
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
COUNTER = 0

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Get eye coordinates
            leftEye = [(int(face_landmarks.landmark[i].x * w),
                        int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            rightEye = [(int(face_landmarks.landmark[i].x * w),
                         int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            # Compute EAR
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eyes
            cv2.polylines(frame, [np.array(leftEye)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(rightEye)], True, (0, 255, 0), 1)

            # Check drowsiness
            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    # winsound.Beep(2500, 1000)  # 🔔 Beep when drowsy
                    playsound("alarm.wav")   # <-- your sound file

            else:
                COUNTER = 0

    cv2.imshow("Drowsiness Detection - Mediapipe", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
