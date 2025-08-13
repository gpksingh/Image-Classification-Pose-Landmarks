#Now that we have the dataset, we use mediapipe to extract the keypoints from images.
import mediapipe as mp

import matplotlib.pyplot as plt
import matplotlib.image as pimg

# Quick test to check if mediapipe is working
# try:
#     mp_drawing = mp.solutions.drawing_utils
#     mp_pose = mp.solutions.pose
#     print("mediapipe imported and modules loaded successfully!")
# except Exception as e:
#     print(f"mediapipe test failed: {e}")


mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.7)

img = pimg.imread('dataset/adho mukha svanasana/1. 5-benefits-of-downward-facing-dog-pose.png')

#
plt.imshow(img)