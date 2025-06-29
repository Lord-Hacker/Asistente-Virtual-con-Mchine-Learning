# Asistente-Virtual-con-Machine-Learning
Trabajo de la asignatura Inteligencia Artificial

Trabajo de Oscar Etcheverry (5647110) y Derlis Delgado (5805739)  

Se utiliza numpy, pandas, pickle, matplotlib, y otras librerías generalmente usadas en python para machine learning  
Además, deben estar instaladas las siguientes librerías de python utilizadas para computer vision:  
opencv, mediapipe, sklearn  

los archivos generate_xxx.py se utilizan para generar el archivo csv, dataset que se utiliza para el entrenamiento
los datasets xxx_train.csv y xxx_test.csv son los datasets ya procesados para usarse para entrenamiento y test respectivavente
xxx_train_mejorado.ipynb se utilizan para entrenar los modelos. Una vez entrenados, generan un archivo pkl
xxx.pkl es el respectivo modelo entrenado, guardado como un archivo, que se puede abrir y utilizar como clase
inferencia_xxx.py se ejecuta cuando se quiere hacer inferencia. Se puede utilizar un video o la webcam
interfaz.py es la interfaz gráfica que utiliza los tres modelos y entrega feedback al ejercitar
