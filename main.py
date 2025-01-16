import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize video capture
cap = cv2.VideoCapture('cctv.mp4')

# Define a function to check if a person is detected in a specific region
def is_person_sitting(pose_landmarks):
    if pose_landmarks:
        # Calculate the angle between hip, knee, and ankle
        left_hip_angle = calculate_angle(pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP], 
                                         pose_landmarks[mp_pose.PoseLandmark.LEFT_KNEE], 
                                         pose_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE])
        
        right_hip_angle = calculate_angle(pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP], 
                                          pose_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE], 
                                          pose_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE])
        
        # Assume person is sitting if the angle is between 70 and 110 degrees
        if 70 <= left_hip_angle <= 110 or 70 <= right_hip_angle <= 110:
            return True
    return False

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

# Placeholders for tracking working and non-working durations
person_data = {}
fps = cap.get(cv2.CAP_PROP_FPS)
frame_skip = 2  # Process every 2nd frame, adjust as needed 

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Convert image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    # Convert back to BGR for visualization
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []

    # Update duration counts based on sitting or standing
    is_sitting = is_person_sitting(pose_landmarks)
    person_id = "person"  # Assuming single person
    if not person_id in person_data:
        person_data[person_id] = {'working_duration': 0, 'non_working_duration': 0}

    if is_sitting:
        person_data[person_id]['working_duration'] += frame_skip / fps
    else:
        person_data[person_id]['non_working_duration'] += frame_skip / fps

    # Draw landmarks for visualization
    mp.solutions.drawing_utils.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('Frame', image)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Generate Summary Table
summary_table = "Person ID | Working Duration (s) | Non-Working Duration (s)\n"
summary_table += "-" * 55 + "\n"
for person_id, data in person_data.items():
    summary_table += f"{person_id} | {data['working_duration']:.2f} | {data['non_working_duration']:.2f}\n"

print(summary_table)
