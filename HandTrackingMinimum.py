# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import time

# # Define hand connections for drawing
# HAND_CONNECTIONS = [
#     (0, 1), (1, 2), (2, 3), (3, 4),
#     (0, 5), (5, 6), (6, 7), (7, 8),
#     (0, 9), (9, 10), (10, 11), (11, 12),
#     (0, 13), (13, 14), (14, 15), (15, 16),
#     (0, 17), (17, 18), (18, 19), (19, 20),
#     (5, 9), (9, 13), (13, 17)
# ]

# cap = cv2.VideoCapture(0)

# model_path = 'hand_landmarker.task'
# base_options = python.BaseOptions(model_asset_path=model_path)
# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.VIDEO,
#     num_hands=2
# )

# with vision.HandLandmarker.create_from_options(options) as landmarker:
#     while True:
#         success, img = cap.read()
#         if not success:
#             break

#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
#         timestamp_ms = int(time.time() * 1000)
#         results = landmarker.detect_for_video(mp_image, timestamp_ms)

#         if results.hand_landmarks:
#             for idx, handLms in enumerate(results.hand_landmarks):
#                 # Draw landmarks
#                 for lm in handLms:
#                     h, w, c = img.shape
#                     cx, cy = int(lm.x * w), int(lm.y * h)
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

#                 # Draw connections
#                 for connection in HAND_CONNECTIONS:
#                     start_idx = connection[0]
#                     end_idx = connection[1]
#                     start_lm = handLms[start_idx]
#                     end_lm = handLms[end_idx]
#                     start_point = (int(start_lm.x * w), int(start_lm.y * h))
#                     end_point = (int(end_lm.x * w), int(end_lm.y * h))
#                     cv2.line(img, start_point, end_point, (0, 255, 0), 2)

#                 # Draw handedness
#                 if results.handedness and idx < len(results.handedness):
#                     handedness = results.handedness[idx][0].category_name
#                     cv2.putText(img, handedness, (10, 70 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         cv2.imshow('Hand Tracking', img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # hands class only uses RGB images
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms)


    cv2.imshow("Image", img) #display the image
    cv2.waitKey(1)