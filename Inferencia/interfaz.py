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

#Importa los modelos
with open('model/count.pkl', 'rb') as f:
  model_count = pickle.load(f)

with open('model/hips.pkl', 'rb') as f:
  model_hips = pickle.load(f)

with open('model/lean.pkl', 'rb') as f:
  model_lean = pickle.load(f)


cap = cv2.VideoCapture("video/test/youtubeA.mp4")
contador = 0
etapa = ''
consejo = ''

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

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=4))


        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks[1:])
            #contador
            body_language_class_count = model_count.predict(X)[0]
            body_language_prob_count = model_count.predict_proba(X)[0]
            #separacion
            body_language_class_hips = model_hips.predict(X)[0]
            body_language_prob_hips = model_hips.predict_proba(X)[0]
            #inclinacion
            body_language_class_lean = model_lean.predict(X)[0]
            body_language_prob_lean = model_lean.predict_proba(X)[0]


            if body_language_class_count == 'down' and body_language_prob_count[body_language_prob_count.argmax()] >= 0.6:
                etapa = 'down'
            elif etapa == 'down' and body_language_class_count == 'up' and body_language_prob_count[body_language_prob_count.argmax()] >= 0.7:
                etapa = 'up'
                contador += 1
            
                
            if body_language_class_lean.split(' ')[0] == 'left':
                consejo = "inclinar a la derecha"
            elif body_language_class_lean.split(' ')[0] == 'right':
                consejo = "inclinar a la izquierda"
            elif body_language_class_hips.split(' ')[0] == 'wide':
                consejo = "cerrar mas las piernas"
            elif body_language_class_hips.split(' ')[0] == 'narrow':
                consejo = "abrir mas las piernas"
            else:
                consejo = "buena tecnica"

            if etapa == 'up':
                incline = body_language_class_lean.split(' ')[0]
                separation = body_language_class_hips.split(' ')[0]
            else:
                incline = "--"
                separation = "--"
                consejo = "arriba"

            
 
            #Elemenetos gráficos para la interfaz
            #Rectángulo de fondo
            cv2.rectangle(image, (0,0), (400,60), (204,136,153), -1)
            cv2.rectangle(image, (image.shape[1]-370,image.shape[0]-50), (image.shape[1]-5,image.shape[0]-5), (204,136,153), -1)

            #CONTADOR
            #Estado (clase en el modelo)
            cv2.putText(image, 'ESTADO', (20,12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class_count.split(' ')[0], (15,40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            #Conteo
            cv2.putText(image, 'REPS', (115,12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(contador), (110,40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 3, cv2.LINE_AA)
             
            #SEPARACION
            #Estado (clase en el modelo)
            cv2.putText(image, 'SEPARACION', (180,12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, separation, (175,40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            #INCLINACION
            #Estado (clase en el modelo)
            cv2.putText(image, 'INCLINACION', (300,12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, incline, (295,40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2, cv2.LINE_AA)


            #CONSEJOS
            cv2.putText(image, consejo, (image.shape[1]-365,image.shape[0]-20), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

            

        except Exception as e:
            print(e)
            #pass

        cv2.imshow('Contador de reps', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
