import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize video capture
cap = cv2.VideoCapture('cctv.mp4')

next_person_id = 0
person_data = {}
fps = cap.get(cv2.CAP_PROP_FPS)
frame_skip = 10  # Process every 2nd frame, adjust as needed

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def is_person_sitting(landmarks):
    if landmarks:
        left_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], 
                                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], 
                                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        right_hip_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], 
                                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], 
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        if 70 <= left_hip_angle <= 110 or 70 <= right_hip_angle <= 110:
            return True
    return False

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.pose_landmarks:
        person_id = next_person_id
        if person_id not in person_data:
            next_person_id += 1
            person_data[person_id] = {'working_duration': 0, 'non_working_duration': 0}

        is_sitting = is_person_sitting(results.pose_landmarks.landmark)
        if is_sitting:
            person_data[person_id]['working_duration'] += frame_skip / fps
        else:
            person_data[person_id]['non_working_duration'] += frame_skip / fps

        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Frame', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

summary_table = "Person ID | Working Duration (s) | Non-Working Duration (s)\n"
summary_table += "-" * 55 + "\n"
for person_id, data in person_data.items():
    summary_table += f"{person_id} | {data['working_duration']:.2f} | {data['non_working_duration']:.2f}\n"

print(summary_table)
