import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    return KMeans, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # La limitación del centroide de K-Means

    Como vimos en la teoría, la función objetivo de K-Means minimiza la Suma de Errores Cuadrados ($SSE$).
    Debido a que la distancia está al cuadrado, un solo valor atípico (Outlier) ejerce una fuerza gravitacional masiva sobre los centroides.

    A continuación, simularemos un conjunto de datos en tiempo real. Se invita a usar el control deslizante para inyectar un outlier y alejarlo progresivamente.
    Observese cómo el centroide abandona a los datos normales para intentar "alcanzar" a la anomalía, deformando la frontera de decisión.
    """)
    return


@app.cell
def _(KMeans, np):
    # Fijamos la semilla para que los clústeres siempre sean iguales
    np.random.seed(42)

    # Generamos 3 clústeres naturales
    C1 = np.random.normal(loc=[-4, 0], scale=[0.8, 0.8], size=(40, 2))
    C2 = np.random.normal(loc=[0, 0], scale=[1.0, 1.0], size=(40, 2))
    C3 = np.random.normal(loc=[4, 0], scale=[0.8, 0.8], size=(40, 2))

    X_limpio = np.vstack([C1, C2, C3])

    # Entrenamos K-Means en los datos limpios para tener nuestra REFERENCIA
    kmeans_ref = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_limpio)
    centroides_ref = kmeans_ref.cluster_centers_
    sse_ref = kmeans_ref.inertia_
    return X_limpio, centroides_ref, sse_ref


@app.cell
def _(KMeans, X_limpio, centroides_ref, np, plt, slider_outlier, sse_ref):
    # 1. Obtenemos el valor actual del slider
    posicion_x = slider_outlier.value

    # 2. Inyectamos el outlier en esa posición
    outlier_movil = np.array([[posicion_x, 0]])
    X_contaminado = np.vstack([X_limpio, outlier_movil])

    # 3. K-Means re-calcula todo en tiempo real
    kmeans_vivo = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_contaminado)
    centroides_vivo = kmeans_vivo.cluster_centers_
    sse_vivo = kmeans_vivo.inertia_

    # 4. Dibujamos la escena
    fig, ax = plt.subplots(figsize=(10, 5))

    # Dibujamos los datos normales (siempre en gris)
    ax.scatter(X_limpio[:, 0], X_limpio[:, 1], c='lightgray', alpha=0.7, s=40, label='Datos Normales')

    # Dibujamos el Outlier interactivo (en rojo)
    ax.scatter(outlier_movil[:, 0], outlier_movil[:, 1], c='red', marker='D', s=120, edgecolors='black', label='Outlier Global')

    # Dibujamos los Centroides ORIGINALES (azul opaco)
    ax.scatter(centroides_ref[:, 0], centroides_ref[:, 1], c='steelblue', marker='x', s=100, alpha=0.4, label='Centroides Originales')

    # Dibujamos los Centroides ACTUALES (cruz roja grande)
    ax.scatter(centroides_vivo[:, 0], centroides_vivo[:, 1], c='darkred', marker='X', s=200, label='Centroides Arrastrados')

    # Formato visual impactante
    ax.set_xlim(-7, 32) # Fijamos el eje X para que no "brinque" al mover el slider
    ax.set_ylim(-4, 4)
    ax.set_title(f"Impacto en Tiempo Real  |  SSE Normal: {sse_ref:.0f}  →  SSE Actual: {sse_vivo:.0f}", fontsize=14, fontweight='bold')
    ax.legend(loc="upper left")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Marimo renderiza la figura simplemente llamándola (o con plt.gca())
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    slider_outlier = mo.ui.slider(
        start=5,      # Posición inicial (cerca de los datos normales)
        stop=30,      # Posición final (muy lejos)
        step=1,       # Movimiento de 1 en 1
        value=5,      # Valor por defecto al abrir el programa
        label="**Magnitud del Outlier (Eje X)**",
        show_value=True
    )

    # Mostramos el slider en pantalla
    slider_outlier
    return (slider_outlier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Detección Contextual con DBSCAN (Smartwatch IoT)

    Analizaremos los datos de un reloj inteligente. Mide dos cosas: **Nivel de Actividad** (0=Durmiendo, 10=Corriendo) y **Ritmo Cardíaco** (PPM).

    * Lo normal es una correlación positiva: a más actividad, más latidos.
    * Utiliza el menú para inyectar una anomalía y juega con los parámetros de **DBSCAN** para aislarla. Observa cómo el algoritmo colorea de rojo (Ruido) a las emergencias que se deben atender.
    """)
    return


@app.cell
def _(dropdown_escenario, np, plt, slider_eps, slider_minpts):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN

    # 1. Generamos los datos del paciente sano (Línea base natural)
    np.random.seed(42)
    actividad_normal = np.random.uniform(0, 10, 150)
    # A mayor actividad, más latidos (con algo de ruido normal)
    ppm_normal = 60 + (actividad_normal * 8) + np.random.normal(0, 8, 150) 

    X_iot = np.column_stack((actividad_normal, ppm_normal))

    # 2. Lógica reactiva: Inyección de Anomalías según el menú
    if dropdown_escenario.value == "2. Fallo de Sensor (Outlier Global)":
        # Falla el reloj: Marca 0 PPM mientras el usuario está caminando (Actividad 5)
        anomalia = np.array([[5.0, 0.0]])
        X_iot = np.vstack([X_iot, anomalia])

    elif dropdown_escenario.value == "3. Taquicardia en Reposo (Outlier Contextual)":
        # Taquicardia: 135 PPM pero el usuario está durmiendo (Actividad 0)
        anomalia = np.array([[0.0, 135.0]])
        X_iot = np.vstack([X_iot, anomalia])

    # 3. Regla de Oro: Escalar los datos antes de usar DBSCAN
    X_iot_escalado = StandardScaler().fit_transform(X_iot)

    # 4. Entrenamos DBSCAN con los valores EN VIVO de los sliders
    dbscan = DBSCAN(eps=slider_eps.value, min_samples=slider_minpts.value)
    etiquetas = dbscan.fit_predict(X_iot_escalado)

    # 5. Separamos los datos para graficarlos (Normales vs Ruido/Anomalías)
    es_normal = etiquetas != -1
    es_anomalia = etiquetas == -1

    # 6. Dibujamos la escena médica
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    # Puntos normales (en azul)
    ax2.scatter(X_iot[es_normal, 0], X_iot[es_normal, 1], 
                c='#1f77b4', alpha=0.7, s=50, edgecolors='white', label='Signos Vitales Normales')

    # Anomalías / Ruido detectado por DBSCAN (en rojo alerta)
    if np.any(es_anomalia):
        ax2.scatter(X_iot[es_anomalia, 0], X_iot[es_anomalia, 1], 
                    c='red', marker='X', s=150, edgecolors='black', label='¡Anomalía Detectada!')

    # Formato de la gráfica médica
    ax2.set_xlim(-1, 11)
    ax2.set_ylim(-10, 160)
    ax2.set_xlabel("Nivel de Actividad (0 = Reposo, 10 = Máximo Esfuerzo)", fontweight='bold')
    ax2.set_ylabel("Ritmo Cardíaco (PPM)", fontweight='bold')

    # Título dinámico
    total_anomalias = np.sum(es_anomalia)
    ax2.set_title(f"Monitorización IoT con DBSCAN  |  Anomalías detectadas: {total_anomalias}", 
                  fontsize=14, fontweight='bold', color='darkred' if total_anomalias > 0 else 'green')
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    # 1. Menú para elegir el tipo de paciente/anomalía
    dropdown_escenario = mo.ui.dropdown(
        options=["1. Persona sana", "2. Fallo de Sensor (Outlier Global)", "3. Taquicardia en Reposo (Outlier Contextual)"],
        value="1. Persona sana",
        label="**Selecciona el Escenario:**"
    )

    # 2. Sliders para calibrar DBSCAN
    slider_eps = mo.ui.slider(
        start=0.1, stop=2.0, step=0.1, value=0.6, 
        label="Radio de Densidad (eps)", show_value=True
    )

    slider_minpts = mo.ui.slider(
        start=2, stop=15, step=1, value=5, 
        label="Masa Crítica (MinPts)", show_value=True
    )

    # Agrupamos los controles en una pila vertical para que se vea como un panel de control
    panel_control = mo.vstack([
        dropdown_escenario,
        mo.md("---"),
        mo.md("**Calibración de DBSCAN:**"),
        slider_eps, 
        slider_minpts
    ])

    panel_control
    return dropdown_escenario, slider_eps, slider_minpts


if __name__ == "__main__":
    app.run()
