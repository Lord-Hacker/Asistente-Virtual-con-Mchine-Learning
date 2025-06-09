# Asistente-Virtual-con-Machine-Learning
Trabajo de la asignatura Inteligencia Artificial

Trabajo de Oscar Etcheverry (5647110) y Derlis Delgado (5805739)  
Link de la presentación:  
https://view.genially.com/6841a0be6f01956893bb2051/presentation-asistente-virtual-de-entrenamiento-de-deadlift-con-machinelearning  

Se utiliza numpy, pandas, pickle, matplotlib, y otras librerías generalmente usadas en python para machine learning  
Además, deben estar instaladas las siguientes librerías de python utilizadas para computer vision:  
opencv, mediapipe, sklearn  

generate_count.py se utiliza para generar el archivo csv, dataset que se utiliza para el entrenamiento  
cords_balances.csv es el dataset ya procesado para tener datos balanceados  
count_train.ipynb se utiliza para entrenar el modelo. Una vez entrenado, genera un archivo pkl  
count.pkl es el modelo entrenado guardado como un archivo, que se puede abrir y utilizar como clase  
inferencia_count.py se ejecuta cuando se quiere hacer inferencia. Se puede utilizar un video o la webcam  
