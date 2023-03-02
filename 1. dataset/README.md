ULG to CSV
-------------------------------

Transformación de ULG a CSV de datos de trayectorias. Se han desarrollado los siguientes scripts.
- CommandsULGtoCSV  : Transforma todos los archivos ULG de un path a múltiples CSV, usando el script ulog2csv(obtenido de pyulog, github).
- GroupCSV          : Agrupa los archivos CSV de una misma trayectoria en un único CSV (de un path con varias trayectorias).
- copydrone         : Busca la línea que contiene el código de modelo de dron y la copia en toda la columna drone_model (a cada archivo de una carpeta).
- print_drone_model : Guarda en un CSV una lista de nombres de trayectorias y su código de modelo de dron a partir de un path con archivos CSV.
- print_model_ulg   : Guarda en un CSV una lista de nombres de trayectorias y su código de modelo de dron a partir de un path con archivos ULG.

Usage
`````
Se debe tener pyulog instalado para ejecutar ulog2csv:
    pip install pyulog

Para ejecutar no hace falta añadir ningún argumento. Lo único que hace falta es definir los paths de las carpetas de entrada y salida de archivos en el código.
El orden de ejecución para transformar trayectorias de ULG a CSV es:
1. commandsULGtoCSV.py
2. GroupCSV.py
3. copydrone.py

print_drone_model y print_model_ulg se ejecutan adicionalmente si se quiere ver y analizar los modelos de dron de las trayectorias de una carpeta sin transformarlos.

Links
`````
* `PX4 Documentation <https://docs.px4.io/main/en/>`_
* `GitHub ulog2csv <https://github.com/PX4/pyulog/blob/main/pyulog/ulog2csv.py>`_