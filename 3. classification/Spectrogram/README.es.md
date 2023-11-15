Procesado de datos y clasificación
-------------------------------
Se han generado los siguientes archivos en google collab y python. Sólo es necesario definir los paths de las carpetas de entrada y salida de archivos en el código.


analisis_modelos.ipynb: Para visualizar en un pie chart la cantidad de cada modelo a partir de la lista de modelos de dron.

analisis_filtrado.ipynb: Contiene pruebas varias para analizar los atributos.

Limpieza_datos.ipynb: Aplica limpieza general de datos/selección de atributos a los datos raw y guarda los datos limpios en una carpeta.

Limpieza_manual.ipynb: Aplica la selección de atributos y limpieza general de los datos de una carpeta, aplicando rdp y limpieza manual en algunos casos. Contiene los métodos utilizados más útiles 

classification.py: Clasificación de los 4 modelos (árbol de decisión, K nearest neighbors, perceptrón y random forest) y print de los resultados de accuracy y fscore.

classification_svm.py: Clasificación del modelo SVM y print de los resultados de accuracy y fscore.


Los siguientes archivos no son importantes, sólo hacen pruebas (guardados en la carpeta "otros programas/pruebas"):

Transformación y rdp.ipynb: Pruebas para la transformación de datos a intervalos y aplicación de RDP a algunas clases.

Transformación de datos+ prueba de clasificación.ipynb: Pruebas iniciales de transformación de datos a intervalos y clasificación.

Modelos de clasificación.ipynb: Pruebas de clasificación con los 5 modelos y algunos tests.

Figuras.ipynb: Para visualizar figuras.