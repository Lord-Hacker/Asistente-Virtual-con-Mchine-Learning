import cv2
import csv
import mediapipe as mp
import os
import numpy as np
from matplotlib import pyplot as plt 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmarks = ['class']
for val in range(1, 33+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

#Guardado de coordenadas de repeticiones
with open('cords_lean.csv', mode = 'w', newline= '') as f:
    csv_writer = csv.writer(f, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

def export_landmark(result, action):
    try:  
        keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        keypoints.insert(0, action)

        with open('cords_lean.csv', mode = 'a', newline = '') as f:
            csv_writer =csv.writer(f, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
            csv_writer.writerow(keypoints)
    except Exception as e:
        pass

cap = cv2.VideoCapture("video/lean/lean_test.mp4")

#inicia la entrada de video
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        #Imagen a formato RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #Hacer detecciones
        results = pose.process(image)

        #Recolorea de vuelta a BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=4))

        #Anotar coordenadas
        k = cv2.waitKey(5)
        
        #Inclinacion
        if k == 122:
            print("RIGHT")
            export_landmark(results, 'right')
        if k == 120:
            print("CENTER")
            export_landmark(results, 'center')
        if k == 99:
            print("LEFT")
            export_landmark(results, 'left')
        
        cv2.imshow('Camara', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Termino")
            break

    cap.release()
    cv2.destroyAllWindows()
