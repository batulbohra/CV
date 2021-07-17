#!/usr/bin/env python
# coding: utf-8

# In[12]:


import time
import math
import cv2
import mediapipe as mp

class hand_tracker():
    def __init__(self, mode=False, max_num_hands=2, detection_conf=0.5, tracking_conf=0.5):
        
        self.mode=mode
        self.max_num_hands=max_num_hands
        self.detection_conf=detection_conf
        self.tracking_conf=tracking_conf
        
        self.hands_mp = mp.solutions.hands
        self.draw_mp = mp.solutions.drawing_utils
        self.hands = self.hands_mp.Hands(mode, max_num_hands, detection_conf, tracking_conf)
        self.tip_ids = [4, 8, 12, 16, 20]
        
    def detection(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.draw_mp.draw_landmarks(frame, hand_lms, self.hands_mp.HAND_CONNECTIONS)
        return frame
    
    def position(self, frame, hand_num=0, draw=True):
        self.lms = []
        lst_cx, lst_cy = [], []
        bbox = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_num]
            for id_num, lm in enumerate(hand.landmark):
                h,w,c = frame.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                self.lms.append([id_num, cx, cy])
                lst_cx.append(cx)
                lst_cy.append(cy)
                if draw:
                    cv2.circle(frame, (cx,cy), 3, (255,0,255), cv2.FILLED)
                    
            minx, miny, maxx, maxy = min(lst_cx), min(lst_cy), max(lst_cx), max(lst_cy)
            bbox = [minx, miny, maxx, maxy]
            if draw:
                cv2.rectangle(frame, (minx - 20, miny - 20), (maxx + 20, maxy + 20),(0, 255, 0), 2)
        return self.lms, bbox    
    
    def fingers_pos(self):
        fingers = []
        #k = self.tip_ids[0]
        #print(self.lms[0])
        if self.lms[self.tip_ids[0]][1] > self.lms[self.tip_ids[0]-1][1] :
            fingers.append(1)
        else:
            fingers.append(0)
        for ids in range(1,5):
            if self.lms[self.tip_ids[ids]][2] < self.lms[self.tip_ids[ids]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    
    def distance(self, p1, p2):
        x1, y1 = self.lms[p1][1:]
        x2, y2 = self.lms[p2][1:]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        
        len = math.hypot(x2-x1, y2-y1)
        return len, [x1, y1, x2, y2, cx, cy]
    

def main():
    cap = cv2.VideoCapture(0)
    timeP = 0
    timeC = 0
    tracker = hand_tracker()
    while True:
        ret, frame = cap.read()
        frame = tracker.detection(frame)
        lm_coord, box = tracker.position(frame)
        if len(lm_coord)!=0:
            print(lm_coord[0])
        #fing = tracker.fingers_pos()
        
        timeC = time.time()
        fps = 1/(timeC-timeP)
        timeP = timeC
        
        cv2.putText(frame,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv2.imshow("Image", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# In[ ]:




