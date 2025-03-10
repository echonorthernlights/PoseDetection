import cv2
import mediapipe as mp
import time


class PoseDetector():
    def __init__(self, static_image_mode = False, model_complexity = 1, smooth_landmarks = True, enable_segmentation = False, smooth_segmentation = True,
                 min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.static_image_mode =static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation,
                                     self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence)


    def find_pose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.pose_landmarks)
        if self.results.pose_landmarks :
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def find_position(self, img, draw=True):
        if self.results.pose_landmarks :
            lmList = []
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                #print(id, lm)
                cy, cx = int(height*lm.y), int(width*lm.x)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)

            return lmList



def main():
    cap = cv2.VideoCapture('SourceMedia/couplewalk.mp4')
    previousTime = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.find_pose(img, draw=False)
        lmList = detector.find_position(img, draw=False)
        # print(lmList[4])
        # cv2.circle(img, (lmList[14][1], lmList[14][2]), 5, (255, 0, 255), cv2.FILLED)
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (100, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()