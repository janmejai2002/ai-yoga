import mediapipe as mp
import numpy as np
mp_pose = mp.solutions.pose
def getLandmarks(landmarks):
    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]    
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

    # # return nose
    return {"nose":nose,
            "right_shoulder":right_shoulder,
            "left_shoulder":left_shoulder,
            "right_wrist":right_wrist,
            "left_wrist":left_wrist,
            "right_elbow":right_elbow,
            "left_elbow":left_elbow,
            "right_hip":right_hip,
            "left_hip":left_hip,
            "right_knee":right_knee,
            "left_knee":left_knee,
            "right_ankle":right_ankle,
            "left_ankle":left_ankle,
            "right_foot_index":right_foot_index,
            "left_foot_index":left_foot_index,
            }
        
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle

    return angle

def getAngles(landmarks):
    nose = landmarks["nose"]
    right_shoulder = landmarks["right_shoulder"]
    left_shoulder = landmarks["left_shoulder"]
    right_wrist = landmarks["right_wrist"]
    left_wrist = landmarks["left_wrist"]
    right_elbow = landmarks["right_elbow"]
    left_elbow = landmarks["left_elbow"]
    right_hip = landmarks["right_hip"]
    left_hip = landmarks["left_hip"]
    right_knee = landmarks["right_knee"]
    left_knee = landmarks["left_knee"]
    right_ankle = landmarks["right_ankle"]
    left_ankle = landmarks["left_ankle"]
    right_foot_index = landmarks["right_foot_index"]
    left_foot_index= landmarks["left_foot_index"]

    
    return {
            "right_shoulder":calculate_angle(right_elbow, right_shoulder, right_hip),
            "left_shoulder":calculate_angle(left_elbow, left_shoulder, left_hip),
            
            "right_shoulder_up":calculate_angle(right_elbow, right_shoulder, left_shoulder),
            "left_shoulder_up":calculate_angle(left_elbow, left_shoulder, right_shoulder),
            
            "right_elbow":calculate_angle(right_wrist, right_elbow, right_shoulder),
            "left_elbow":calculate_angle(left_wrist, left_elbow, left_shoulder),

            "right_hip_in":calculate_angle(left_hip, right_hip, right_knee),
            "left_hip_in":calculate_angle(right_hip, left_hip, left_knee),

            "right_hip_out":calculate_angle(right_knee, right_hip, right_shoulder),
            "left_hip_out":calculate_angle(left_knee, left_hip, left_shoulder),

            "right_knee":calculate_angle(right_ankle, right_knee, right_hip),
            "left_knee":calculate_angle(left_ankle, left_knee, left_hip),
            
            }

headers = ['class',
           'image_id',
           'right_shoulder',
           'left_shoulder',
           'right_shoulder_up',
           'left_shoulder_up',
           'right_elbow',
           'left_elbow',
           'right_hip_in',
           'left_hip_in',
           'right_hip_out',
           'left_hip_out',
           'right_knee',
           'left_knee',]
