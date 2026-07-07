# tfm-unir-manuel-antonio-fontoira-colorado
Código en python y dataset utilizados para el desarrollo experimental del TFM.
TFM: Modelo de clasificación predictivo para detección de supernovas
Descripción

Este repositorio contiene el código y los datos utilizados para el desarrollo del Trabajo Fin de Máster (TFM). Su objetivo es facilitar la reproducibilidad de los experimentos y de los resultados presentados en la memoria.

Estructura del repositorio
.
├── src/
│   └── main.py
├── data/
│   ├── raw/
│   │   ├── metadata.csv
│   │   └── lightcurves.csv
│   └── processed/
│       └── dataset_final.csv
├── results/
│   └── comparativa_modelos.txt
└── README.md
src/: código fuente empleado para el preprocesamiento, entrenamiento y evaluación de los modelos.
data/raw/: datos originales utilizados como punto de partida.
data/processed/: conjunto de datos final tras el preprocesamiento.
results/: resultados finales de comparación de modelos.
Requisitos

El proyecto ha sido desarrollado en Python.

Las principales dependencias pueden instalarse mediante:

pip install -r requirements.txt
Ejecución

Desde la raíz del repositorio:

python src/main.py

El script genera los resultados experimentales descritos en la memoria del TFM.

Relación con el TFM

Este repositorio complementa la memoria del Trabajo Fin de Máster y contiene el código y los datos empleados para obtener los resultados experimentales presentados.

Autor

Manuel Antonio Fontoira Colorado
