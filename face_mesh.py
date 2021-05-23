# -*- coding: utf-8 -*-
"""
Created on Sun May 23 13:58:13 2021

@author: shri
"""

import mediapipe as mp
import cv2 
import numpy as np


cap = cv2.VideoCapture(0)     ### used to open a connection with local webcam
facemesh = mp.solutions.face_mesh           
face = facemesh.FaceMesh(static_image_mode= True , min_tracking_confidence=0.6 , min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils


while True:
    
    _ , frame = cap.read()                  ##reading from local_Web_cam
    frame = cv2.flip(frame , 1)               ### 1 for flipping frame horizontally
    rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)     ### cv2 only workg on RGB so converting frame to RGB
    op = face.process(rgb)                         ### generating face mask
    if op.multi_face_landmarks:                ## looping the face mask points
       for i in op.multi_face_landmarks:
           draw.draw_landmarks( frame , i , facemesh.FACE_CONNECTIONS , landmark_drawing_spec=draw.DrawingSpec(color=(249 , 249 , 8) , circle_radius = 1))           
           #### drawing on the landmarks returned by face mesh
    cv2.imshow("image" , frame)
    
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
    
