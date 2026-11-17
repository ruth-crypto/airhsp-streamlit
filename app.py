import json
import pandas as pd
import streamlit as st
import plotly.express as px
from motor import BASE, cargar_modelos, decisiones, predecir, proyectar

st.set_page_config(page_title='AIRHSP | Observatorio de perfiles', page_icon='📊', layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem;max-width:1250px}h1{letter-spacing:-1px}div[data-testid="stMetric"]{background:#eef5f7;padding:18px;border-radius:12px;border-left:4px solid #087f8c}section[data-testid="stSidebar"]{background:#edf3f5}</style>''', unsafe_allow_html=True)
@st.cache_data
def tabla(nombre):
    return pd.read_csv(BASE / nombre)
@st.cache_resource
def modelos():
    return cargar_modelos()
@st.cache_data
def mapa_pca():
    datos = tabla('resultados_no_supervisado/perfiles_clusters.csv')
    muestra = datos.sample(min(3500, len(datos)), random_state=42).copy()
    _, seg = modelos()
    Z, grupos = proyectar(seg, muestra)
    return pd.DataFrame({'PC1':Z[:,0], 'PC2':Z[:,1], 'Grupo':(grupos+1).astype(str), 'Sector':muestra.SECTOR.to_numpy()})

st.sidebar.title('AIRHSP')
st.sidebar.caption('Applied Machine Learning · UTEC')
pagina = st.sidebar.radio('Explora el proyecto', ['Conoce el proyecto','Modelos evaluados','Resultados y aprendizajes','Explora los grupos','Realiza una predicción'])
st.sidebar.divider()
st.sidebar.caption('Prototipo académico. Unidad: perfil institucional-laboral por mes. No representa una persona ni una plaza individual.')
st.title(pagina)
MESES = ['enero','febrero','marzo','abril','mayo','junio','julio','agosto','septiembre','octubre','noviembre','diciembre']
def mes_legible(valor):
    f = pd.to_datetime(str(int(valor)), format='%Y%m%d')
    return f'{MESES[f.month-1]} de {f.year}'


if pagina == 'Conoce el proyecto':
    st.subheader('¿Qué podemos aprender de los registros del empleo público?')
    st.write('Este proyecto está pensado como una demostración para analistas que revisan información del empleo público. Permite conocer los resultados del estudio y probar una estimación del estado de un perfil en el mes siguiente.')
    with st.container(border=True):
        st.markdown('### Antes de comenzar: ¿qué es un perfil?')
        st.write('Es una agrupación de registros que comparten características, como la institución, la unidad ejecutora, el régimen laboral y el cargo. No es la ficha de una persona.')
        st.markdown('**Ejemplo ilustrativo:** los registros de un mismo cargo y régimen laboral dentro de una unidad de una institución pueden formar un perfil. En cada mes observamos si predomina el estado OCUPADO o VACANTE.')
        st.caption('La app estima ese estado predominante para el siguiente mes. No calcula cuántas plazas quedarán vacantes ni quién dejará su trabajo.')
    a,b,c = st.columns(3)
    a.markdown('**1 · Entiende**\n\nConoce el problema y los modelos que comparamos.')
    b.markdown('**2 · Explora**\n\nRevisa los hallazgos y los grupos de perfiles.')
    c.markdown('**3 · Prueba**\n\nSelecciona un ejemplo y obtén una estimación.')
    st.write('Utilizamos datos AIRHSP para estimar si el estado dominante de un perfil institucional-laboral será OCUPADO o VACANTE en el siguiente mes consecutivo. La herramienta permite explorar perfiles y apoyar su revisión; no determina decisiones de contratación.')
    a,b,c = st.columns(3)
    a.metric('Modelos supervisados', '6 familias')
    b.metric('Horizonte', '1 mes')
    c.metric('Segmentos descriptivos', '8 grupos')
    st.markdown('### ¿Cómo construimos la solución?')
    st.write('Organizamos los datos por mes, construimos los perfiles y comparamos lo que ocurría en un mes con el siguiente. Entrenamos varios métodos con meses anteriores y los evaluamos con meses posteriores.')
    st.info('Un perfil reúne características institucionales y laborales compartidas. Una predicción de VACANTE se refiere a su estado dominante; no al número de vacantes ni al riesgo de salida de un trabajador.')
    st.markdown('### Qué encontramos')
    st.write('La clase VACANTE es minoritaria. El estado actual aporta mucha información, por eso comparamos también con persistencia: asumir que el próximo mes conserva el estado actual.')
    st.write('La evaluación es retrospectiva. Un registro extremo de agosto requiere revisión en la fuente y limita las conclusiones. Hace falta una validación prospectiva antes de uso operativo.')
    st.success('Para la demo: abre «Realiza una predicción», elige un perfil y pulsa «Estimar estado del próximo mes».')

elif pagina == 'Modelos evaluados':
    familias = {'Regresión logística':'Referencia lineal interpretable.', 'Árbol de decisión':'Reglas no lineales.', 'Random Forest':'Combinación de árboles.', 'XGBoost':'Boosting de árboles.', 'LightGBM':'Boosting eficiente; modelo seleccionado.', 'CatBoost':'Otra alternativa de boosting.'}
    st.dataframe(pd.DataFrame(familias.items(), columns=['Modelo','Papel en la comparación']), hide_index=True, width='stretch')
    st.write('Referencias: Dummy y persistencia. El blend se conserva como diagnóstico adicional; no es el modelo desplegado.')
    st.subheader('¿Por qué elegimos LightGBM?')
    st.write('Los modelos son métodos que aprenden patrones a partir de ejemplos. LightGBM combina árboles de decisión: pequeñas reglas que, juntas, producen una puntuación para cada perfil.')
    st.write('Seleccionado por Average Precision (AP) en validación temporal. El umbral principal maximiza F1 en validación. La aplicación reutiliza el pipeline entrenado, sin reentrenarlo ni ajustar umbrales con el test.')
    st.info('AP resume la capacidad de ordenar la clase VACANTE. No es exactitud ni porcentaje de aciertos. En algunos archivos originales figura como PR_AUC_val.')
    with st.expander('Detalle técnico: ajustes de los modelos'):
        st.dataframe(tabla('tuning_base_vs_ajustado.csv'), hide_index=True, width='stretch')

elif pagina == 'Resultados y aprendizajes':
    st.write('Comparamos los métodos para saber qué aportan y dónde se equivocan. Una puntuación alta en una métrica no significa ser mejor en todos los aspectos.')
    st.info('Validación: datos usados para elegir el modelo. Test retrospectivo: datos posteriores usados para revisar su desempeño, con las limitaciones históricas explicadas abajo.')
    origen = st.radio('Conjunto de evaluación', ['Validación','Test retrospectivo'], horizontal=True)
    val = origen == 'Validación'
    t = tabla('resultados_validacion.csv' if val else 'resultados_test_retrospectivo.csv')
    metrica = 'PR_AUC_val' if val else 'AP_test'
    fig = px.bar(t.sort_values(metrica), x=metrica, y='modelo', orientation='h', labels={metrica:'Average Precision (AP)','modelo':'Modelo'}, color_discrete_sequence=['#087f8c'])
    fig.update_layout(height=550)
    st.plotly_chart(fig, width='stretch')
    with st.expander('Ver tabla completa de métricas'):
        st.dataframe(t, hide_index=True, width='stretch')
    st.download_button('Descargar comparativa', t.to_csv(index=False).encode('utf-8-sig'), 'comparativa.csv', 'text/csv')
    st.caption('AP mide qué tan bien ordena el modelo los casos VACANTE. Va de 0 a 1; un valor mayor es mejor. No es el porcentaje de aciertos.')
    st.subheader('Tres ideas para llevarte')
    st.markdown('**1. Podemos ordenar los perfiles.** LightGBM produce una puntuación que ayuda a priorizar su revisión.\n\n**2. Una regla simple sigue siendo competitiva.** Persistencia significa suponer que el próximo mes se mantiene el estado actual.\n\n**3. Todavía falta validar su uso real.** Necesitamos nuevos meses y costos acordados con quienes tomarían las decisiones.')
    st.write('LightGBM ajustado aporta una puntuación para ordenar perfiles por señal de VACANTE. Su selección se hizo en validación; no implica superioridad en todas las métricas.')
    test = tabla('resultados_test_retrospectivo.csv').set_index('modelo')
    elegido = decisiones()['campeon']
    if elegido in test.index and 'Persistencia' in test.index:
        a,b,c = st.columns(3)
        a.metric('AP · LightGBM', f"{test.loc[elegido,'AP_test']:.4f}")
        b.metric('F1 · LightGBM', f"{test.loc[elegido,'f1']:.4f}")
        c.metric('F1 · Persistencia', f"{test.loc['Persistencia','f1']:.4f}")
    st.warning('Persistencia supera a LightGBM en F1 y en el escenario de costo evaluado en test. No presentamos el prototipo como una mejora operativa demostrada. Los costos FN=100 y FP=5 son supuestos de experimentación, no montos monetarios verificados.')
    st.caption('F1 resume el equilibrio entre detectar casos VACANTE y evitar falsas alertas. Mayor es mejor; no equivale a exactitud.')
    st.subheader('Profundiza en los experimentos (opcional)')
    experimento = st.selectbox('Comparativa', ['Ajuste de hiperparámetros','Desbalance','Umbrales','Ablación de variables','Calibración'])
    rutas = ['tuning_base_vs_ajustado.csv','exp1_desbalance_validacion.csv','exp3_umbrales_validacion.csv','exp4_ablacion_validacion.csv','calibracion_validacion.csv']
    st.dataframe(tabla(rutas[['Ajuste de hiperparámetros','Desbalance','Umbrales','Ablación de variables','Calibración'].index(experimento)]), hide_index=True, width='stretch')
    st.markdown('**Conclusión:** el modelo permite priorizar una revisión, pero necesitamos validar con nuevos meses, revisar anomalías de la fuente y acordar costos reales antes de decidir si reemplaza a persistencia.')

elif pagina == 'Explora los grupos':
    st.info('Un grupo reúne perfiles con características económicas y de tamaño parecidas. Sirve para describir diferencias entre ellos; no indica cuál quedará VACANTE.')
    st.write('K-means agrupa perfiles por variables económicas y tamaño, sin utilizar estado ni etiqueta futura en el ajuste. Corte: enero de 2026. Los números de grupo no indican mayor o menor riesgo.')
    tamanos = tabla('resultados_no_supervisado/tamanos_clusters.csv')
    a,b = st.columns(2)
    grafico_tamanos = tamanos.assign(CLUSTER_KMEANS=tamanos.CLUSTER_KMEANS+1)
    with a:
        st.plotly_chart(px.bar(grafico_tamanos, x='CLUSTER_KMEANS', y='perfiles', labels={'CLUSTER_KMEANS':'Grupo','perfiles':'Perfiles'}, color_discrete_sequence=['#087f8c']), width='stretch')
    with b:
        st.plotly_chart(px.scatter(mapa_pca(), x='PC1', y='PC2', color='Grupo', hover_data=['Sector'], opacity=.55), width='stretch')
    st.caption('PCA: proyección de una muestra reproducible de hasta 3 500 perfiles. PC1 y PC2 explican 42.20% de la varianza; el ajuste usa siete componentes (91.25%).')
    g = st.selectbox('Elige un grupo para conocerlo', tamanos.CLUSTER_KMEANS.tolist(), format_func=lambda x: f'Grupo {x + 1}')
    med = tabla('resultados_no_supervisado/medianas_clusters.csv')
    fila_grupo = tamanos[tamanos.CLUSTER_KMEANS == g].iloc[0]
    st.write(f"Este grupo reúne **{int(fila_grupo.perfiles):,} perfiles**, el **{fila_grupo.porcentaje:.1f}%** del corte analizado.")
    composicion = tabla('resultados_no_supervisado/composicion_DESC_REGIMEN_LABORAL.csv').set_index('CLUSTER_KMEANS').loc[g]
    dominante = composicion.astype(float).idxmax()
    st.write(f'El régimen laboral más frecuente es **{dominante}** ({float(composicion[dominante]):.1f}% del grupo). Esta característica describe el grupo; no se usó para construirlo.')
    st.caption('Mediana: valor que deja la mitad de los perfiles por debajo y la mitad por encima.')
    with st.expander('Ver características económicas del grupo'):
        st.dataframe(med[med.CLUSTER_KMEANS == g].set_index('CLUSTER_KMEANS').T.rename(columns={g:'Mediana'}), width='stretch')
    st.subheader('Comparación sobre la misma muestra')
    st.dataframe(tabla('resultados_no_supervisado/comparacion_muestra.csv'), hide_index=True, width='stretch')
    st.caption('DBSCAN excluye 22.96% de ruido para calcular sus métricas. Su silhouette no debe interpretarse como superioridad directa sobre la cobertura completa de K-means.')

else:
    modelo, segmento = modelos()
    df = tabla('ejemplos_prototipo.csv')
    st.write('Prueba un caso histórico en tres pasos. La aplicación carga los datos del perfil y usa el modelo entrenado para estimar el estado del mes siguiente.')
    st.caption('Demostración con 300 ejemplos históricos. Las instituciones disponibles corresponden a esta muestra; no es una consulta actualizada de todo el empleo público.')
    st.subheader('1. Elige el perfil que quieres revisar')
    a,b = st.columns(2)
    with a:
        sector = st.selectbox('Sector', sorted(df.SECTOR.dropna().unique()), help='Ámbito al que pertenece la institución.')
        disponibles = df[df.SECTOR == sector]
        pliego = st.selectbox('Institución (pliego)', sorted(disponibles.PLIEGO.dropna().unique()), help='Entidad presupuestaria registrada en AIRHSP.')
    with b:
        disponibles = disponibles[disponibles.PLIEGO == pliego]
        unidad = st.selectbox('Unidad ejecutora', sorted(disponibles.UNIDAD_EJECUTORA.dropna().unique()), help='Unidad de la institución a la que pertenecen los registros.')
        disponibles = disponibles[disponibles.UNIDAD_EJECUTORA == unidad]
        idx = st.selectbox('Cargo, régimen y mes del ejemplo', disponibles.index.tolist(), format_func=lambda i: f"{df.loc[i,'DESC_CARGO_ESTRUCUTURAL']} · {df.loc[i,'DESC_REGIMEN_LABORAL']} · {mes_legible(df.loc[i,'PERIODO'])} · ejemplo {i+1}")
    fila = df.loc[[idx]].copy()
    original = fila.iloc[0]
    st.subheader('2. Revisa los datos del ejemplo')
    with st.container(border=True):
        st.markdown(f"**Cargo:** {original.DESC_CARGO_ESTRUCUTURAL}")
        st.write(f"Régimen: {original.DESC_REGIMEN_LABORAL}")
        a,b = st.columns(2)
        a.metric('Estado predominante observado', original.ESTADO_REGISTRO)
        b.metric('Mes de origen', mes_legible(original.PERIODO).capitalize())
        st.caption('Un perfil agrupa registros con características compartidas. Su estado predominante no describe a una persona específica.')
    st.subheader('3. Obtén la estimación')
    modo = st.radio('¿Qué quieres probar?', ['Usar los datos originales','Explorar un escenario hipotético'], key=f'modo_{idx}')
    with st.form(f'prediccion_{idx}'):
        estado = original.ESTADO_REGISTRO
        if modo == 'Explorar un escenario hipotético':
            estado = st.selectbox('Estado actual para la simulación', ['OCUPADO','VACANTE'], index=int(estado == 'VACANTE'))
            st.caption('Solo cambiaremos el estado actual. El resto de las variables conservará sus valores. Esto no demuestra el efecto causal de cambiar el estado.')
        with st.expander('Opciones avanzadas (opcional)'):
            politica = st.selectbox('Cómo convertir la puntuación en una clasificación', ['Máx. F1','Mínimo costo'], format_func=lambda x: {'Máx. F1':'Equilibrio entre detección y falsas alertas','Mínimo costo':'Escenario de costos del experimento'}[x])
            st.write('La opción predeterminada usa el umbral de máximo F1 elegido en validación. La alternativa asigna un costo supuesto de 100 a una omisión y 5 a una falsa alerta. No son costos monetarios verificados.')
        enviar = st.form_submit_button('Estimar estado del próximo mes', type='primary')
    entrada = fila[modelo.named_steps['prep'].cols_].copy()
    entrada['ESTADO_REGISTRO'] = estado
    with st.expander('Consultar todas las variables utilizadas'):
        st.caption('Estos datos se cargan automáticamente. La etiqueta del mes siguiente no es una entrada del modelo.')
        st.dataframe(entrada.T.rename(columns={idx:'Valor'}).astype(str), width='stretch')
    if enviar:
        fila['ESTADO_REGISTRO'] = estado
        umbral = decisiones()['criterios'][politica]
        r = predecir(modelo, fila, umbral).iloc[0]
        fecha = pd.to_datetime(str(int(original.PERIODO)), format='%Y%m%d')
        siguiente = fecha + pd.DateOffset(months=1)
        mes_destino = f'{MESES[siguiente.month-1]} de {siguiente.year}'
        st.divider()
        st.subheader('Resultado de la estimación')
        st.info(f'Para {mes_destino}, el modelo estima que el estado predominante de este perfil será **{r.estado_predicho}**.')
        st.write('Esto se refiere al conjunto de registros del perfil. No significa que todas sus plazas tengan ese estado ni predice la salida de una persona.')
        a,b = st.columns(2)
        score_texto = '< 0.01%' if 0 < r.score_vacante < .0001 else f'{r.score_vacante:.2%}'
        a.metric('Puntuación del modelo para VACANTE', score_texto)
        b.metric('Si conserváramos el estado actual', estado)
        st.caption('La puntuación expresa la señal que asigna el modelo a VACANTE. No es una certeza; se compara con un umbral para obtener la clasificación.')
        st.markdown('**¿Cómo utilizarlo?** Como una señal para revisar el perfil junto con sus registros y contexto. Por sí solo, el resultado no justifica una decisión de contratación ni demuestra que mejore la regla de conservar el estado actual.')
        if modo == 'Explorar un escenario hipotético':
            st.warning('Resultado de una simulación: estás explorando una entrada hipotética, no una observación nueva.')
        else:
            with st.expander('¿Qué ocurrió después en los datos históricos?'):
                st.write('Estado observado: **'+('VACANTE' if int(original.TARGET) == 1 else 'OCUPADO')+'**.')
                st.caption('Se muestra después de predecir, solo para interpretar este ejemplo. Un caso aislado no mide el desempeño global.')
        with st.expander('Detalle técnico del resultado'):
            st.write(f'Modelo: {decisiones()["campeon"]}. Criterio: {politica}. Umbral: {umbral:.6f}.')
        salida = pd.DataFrame([{'periodo_origen':int(original.PERIODO),'mes_predicho':siguiente.strftime('%Y-%m'),'perfil':original.PERFIL_ID,'modo':modo,'criterio':politica,'umbral':umbral,**r.to_dict()}])
        st.download_button('Descargar esta estimación', salida.to_csv(index=False).encode('utf-8-sig'), 'prediccion_airhsp.csv', 'text/csv')
