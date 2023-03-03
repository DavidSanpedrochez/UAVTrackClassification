[![](https://img.shields.io/badge/README-Español-red)](./README.es.md)
# English version:
This repository contains the source code for a line of investigation centered in the aplication of machine learning approaches over UAV trajectories
the preprint: ***UAV airframe classification based on trajectory data in UTM collaborative environments***, made by David Sánchez Pedroche, Daniel Amigo, Jesús García and José Manuel Molina, from University Carlos III of Madrid.





---------

## Original data

    
#### Dependencies and versions used:

  
## Using the illustration generation framework


#### Dependencies and versions used:


## Contact
If you have any questions, suggestions or problems with the code or the item, please do not hesitate to contact damigo@inf.uc3m.es or davsanch@inf.uc3m.es.




# Spanish version:
Este repositorio contiene el código utilizado para diversos estudios de mineria de datos, hecho por David Sánchez Pedroche, Daniel Amigo, Jesús García y José Manuel Molina, de la Universidad Carlos III de Madrid.

Se incluye la referencia en formato BibTeX. Si le ha servido este artículo o este código, cítenos.
El artículo está disponible en: https://www.researchgate.net/publication/342723317_Architecture_for_Trajectory-Based_Fishing_Ship_Classification_with_AIS_Data
También se proporciona el código fuente para generar las gráficas realizadas en el artículo.

## Datos originales
Los datos utilizados son propiedad de Danish Maritime Authority, puestos a disposición de cualquiera en:
https://dma.dk/SikkerhedTilSoes/Sejladsinformation/AIS/Sider/default.aspx (enlace en: *Get historical AIS data*).
El artículo utiliza tres días del mes de Junio de 2017: *dk_csv_jun2017.rar*. En concreto: *aisdk_20170625.csv, aisdk_20170626.csv y aisdk_20170627.csv*.
El resto de datasets se obtienen con la ejecución de estos datos originales y el código proporcionado.

## Uso del framework de MATLAB
  Es necesario añadir todo el framework al path de Matlab. Para ello:
  
  ```
  - Home
  - Set Path
  - Add with Subfolders...
  - Root folder
  - Ok
  - Save
  ```
  
  El fichero **script.m** orquesta toda la ejecución, desde los ficheros originales de cada día, realiza todo el limpiado, filtrado y preprocesado, y finalmente la clasificación. Cada uno de los pasos es customizable según los parámetros que se pasen a la función. A grandes rasgos (para más detalle leer el artículo) el proceso completo ejecuta las siguientes funciones:
  
    1. mmsiToCleaned(selectedDay, cleanedParameters)
    2. cleanedToTimestamp(selectedDay, timestampParameters)
    3. timestampFolderToFilter(selectedDay, filterParameters)
    4. mergeDays(days)
    5. segmentation(segmentationAlgorithm)
    6. extractSegmentInfo(segmentationAlgorithm)
    7. classifier(segmentationAlgorithm, classificationParameters)  
    
#### Dependencias y versiones utilizadas:
El código ha sido probado en la versión MATLAB R2020 Update 1, 64-bit (win64). Requiere el uso de la Toolbox Parallel Computing. Se puede desactivar si se cambian los bucles "parfor" por "for".
  
## Uso del framework de generación de gráficas
Para realizar las gráficas se utiliza Python + Jupyter Notebook + Plot.ly, ejecutado desde VSCode. También se proporciona el código base y datos de las mismas.

#### Dependencias y versiones utilizadas:
Se requiere que el fichero *ipynb* pueda acceder a los datos *csv* (mediante la bibilioteca *Pandas*).
Ha sido probado con Python 3.7.6 64-bit mediante Anaconda 3 (en VSCode requiere la extensión *ms-python.anaconda-extension-pack*).

## Contacto
Cualquier duda, sugerencia o problema con el código o el artículo, no duden en contactar con damigo@inf.uc3m.es o davsanch@inf.uc3m.es