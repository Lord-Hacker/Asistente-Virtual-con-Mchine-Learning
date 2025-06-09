import mediapipe as mp
import cv2

import pickle
import pandas as pd
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmarks = ['class']
for val in range(1, 33+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

#Importa la clase
with open('count.pkl', 'rb') as f:
  model = pickle.load(f)

cap = cv2.VideoCapture(0)
contador = 0
etapa = ''

#inicia hollistic
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        #Imagen a formato RGB
        image = frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #Hacer detecciones
        results = pose.process(image)

        #Recolorea de vuelta a BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Dibuja las detecciones
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=4))



        try:
            #inferencia
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks[1:])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]

            #contador de repeticiones
            if body_language_class == 'down' and body_language_prob[body_language_prob.argmax()] >= 0.6:
                etapa = 'down'
            elif etapa == 'down' and body_language_class == 'up' and body_language_prob[body_language_prob.argmax()] >= 0.7:
                etapa = 'up'
                contador += 1
            
            #Elemenetos gráficos para la interfaz
            #Rectángulo de fondo
            cv2.rectangle(image, (0,0), (250,60), (204,136,153), -1)

            #Estado (clase en el modelo)
            cv2.putText(image, 'ESTADO', (95,12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            #Probabilidad
            cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            #Conteo
            cv2.putText(image, 'REPS', (200,12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(contador), (195,40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        except Exception as e:
            pass

        cv2.imshow('Contador de reps', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
