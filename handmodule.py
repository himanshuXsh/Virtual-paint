import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.pTime = 0

    def process(self, img):
        """Process the image and detect hands."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        return self.results

    def drawHandLandmarks(self, img, results):
        """Draw hand landmarks and connections on the image."""
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # Optional: draw circles on all landmarks
                    # cv2.circle(img, (cx, cy), 8, (255, 1, 255), cv2.FILLED)
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """Find hand landmark positions."""
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList
    def fingersUp(self):
        """
        Returns a list of 5 integers (1 or 0) representing which fingers are up.
        1 = finger is up, 0 = finger is down
        Thumb, Index, Middle, Ring, Pinky
        """
        fingers = []
        tipIds = [4, 8, 12, 16, 20]  # Landmark indices for fingertip points

        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            h, w = 720, 1280  # adjust based on your image size if needed

            # Get landmark positions
            lmList = []
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            # Thumb (compare x for thumb)
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other 4 fingers (compare y)
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers


    def getFPS(self):
        """Calculate and return frames per second."""
        cTime = time.time()
        fps = 1 / (cTime - self.pTime) if cTime != self.pTime else 0
        self.pTime = cTime
        return int(fps)

