import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
# static_image_mode = False,
# model_complexity = 1,
# smooth_landmarks = True,
# enable_segmentation = False,
# smooth_segmentation = True,
# min_detection_confidence = 0.5,
# min_tracking_confidence = 0.5

pose = mpPose.Pose()

cap = cv2.VideoCapture('SourceMedia/couplewalk.mp4')
previousTime = 0
currentTime = 0
while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks :
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            #print(id, lm)
            cy, cx = int(height*lm.y), int(width*lm.x)
            cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)
            print(id, cx,cy)
    currentTime = time.time()

    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (100,100), cv2.FONT_HERSHEY_DUPLEX, 3,(255,0,255),3 )
    cv2.imshow("Image", img)
    cv2.waitKey(1)