from pathlib import Path
import json
import sys
import joblib
import numpy as np
import pandas as pd
from preparacion import PreparadorAIRHSP

BASE = Path(__file__).resolve().parent / 'data'

def cargar_modelos():
    # Compatibilidad con la clase serializada desde una celda de Colab.
    sys.modules['__main__'].PreparadorAIRHSP = PreparadorAIRHSP
    modelo = joblib.load(BASE / 'modelo_campeon_pipeline.joblib')
    segmento = joblib.load(BASE / 'resultados_no_supervisado/segmentacion_kmeans.joblib')
    return modelo, segmento

def decisiones():
    return json.loads((BASE / 'decisiones_antes_test.json').read_text(encoding='utf-8'))

def predecir(modelo, datos, umbral):
    cols = modelo.named_steps['prep'].cols_
    faltan = set(cols) - set(datos.columns)
    if faltan:
        raise ValueError('Faltan variables: ' + ', '.join(sorted(faltan)))
    X = datos[cols].copy()
    clase = list(modelo.classes_).index(1)
    scores = modelo.predict_proba(X)[:, clase]
    return pd.DataFrame({'score_vacante': scores, 'estado_predicho': np.where(scores >= umbral, 'VACANTE', 'OCUPADO')}, index=datos.index)

def proyectar(segmento, datos):
    X = datos[segmento['variables']].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan)
    X = X.fillna(segmento['medianas'])
    if (X < 0).any().any():
        raise ValueError('La segmentación requiere valores no negativos.')
    if segmento['log1p']:
        X = np.log1p(X)
    Z = segmento['pca'].transform(segmento['scaler'].transform(X))
    return Z, segmento['kmeans'].predict(Z)
