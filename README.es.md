[![](https://img.shields.io/badge/README-English-blue)](./README.md)


# Spanish version:
Este repositorio contiene el código utilizado para diversos estudios de mineria de datos, hecho por David Sánchez Pedroche, Daniel Amigo, Jesús García y José Manuel Molina, de la Universidad Carlos III de Madrid.

Se incluye la referencia en formato BibTeX. Si le ha servido este artículo o este código, cítenos.
El artículo está disponible en: https://www.researchgate.net/publication/342723317_Architecture_for_Trajectory-Based_Fishing_Ship_Classification_with_AIS_Data
También se proporciona el código fuente para generar las gráficas realizadas en el artículo.

## Datos originales
Los datos utilizados son propiedad de 
El resto de datasets se obtienen con la ejecución de estos datos originales y el código proporcionado.


- `UAVTrackClassification`: Directorio con todo el código fuente.
    - [![](https://img.shields.io/badge/README-1. dataset-red)](./README.es.md): Componentes para la generación del dataset.
    - [![](https://img.shields.io/badge/README-2. data preparation-red)](./README.es.md): Componentes para la preparación de las entradas de los algoritmos de clasificacion.
    - [![](https://img.shields.io/badge/README-3. dataset-red)](./README.es.md): Componentes para la aplicación de algoritmos de clasificacion.
    
#### Dependencias y versiones utilizadas:
El código ha sido probado en la versión MATLAB R2020 Update 1, 64-bit (win64). Requiere el uso de la Toolbox Parallel Computing. Se puede desactivar si se cambian los bucles "parfor" por "for".
  
## Uso del framework de generación de gráficas
Para realizar las gráficas se utiliza Python + Jupyter Notebook + Plot.ly, ejecutado desde VSCode. También se proporciona el código base y datos de las mismas.

#### Dependencias y versiones utilizadas:
Se requiere que el fichero *ipynb* pueda acceder a los datos *csv* (mediante la bibilioteca *Pandas*).
Ha sido probado con Python 3.7.6 64-bit mediante Anaconda 3 (en VSCode requiere la extensión *ms-python.anaconda-extension-pack*).

## Contacto
Cualquier duda, sugerencia o problema con el código o el artículo, no duden en contactar con damigo@inf.uc3m.es o davsanch@inf.uc3m.es