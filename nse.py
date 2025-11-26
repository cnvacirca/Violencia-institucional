import json
import pandas as pd

df = pd.read_csv("Encuesta.csv")

# Rutas de entrada/salida
ENCUESTA_LIMPIA = "Encuesta_limpia.csv"
CONFIG_PUNTAJE = "config_nse.json"
ENCUESTA_CON_NSE = "Encuesta_limpia.csv" # Para que sobreescriba el archivo

# Columnas fuente
COLUMNA_BARRIO = df.columns[df.columns.str.startswith("4")][0]
COLUMNA_EDUCACION = df.columns[df.columns.str.startswith("5")][0]
COLUMNA_OCUPACION = df.columns[df.columns.str.startswith("6")][0]
COLUMNA_TRABAJO = df.columns[df.columns.str.startswith("7")][0]

COLUMNA_SALIDA_NSE = "nivel socioeconómico"
COLUMNA_PERCENTIL_NSE = "percentil NSE"

def cargar_configuracion(ruta_config):
    with open(ruta_config, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config

def puntuar_serie_desde_mapeo(serie_texto, mapeo):
    # Mapea 1:1 contra las claves del JSON.
    return serie_texto.map(mapeo).astype(float)


# Lleva una Serie numérica a escala [0, 1] usando normalización min–max.
# - El valor mínimo pasa a 0, el máximo a 1 y el resto queda proporcional.
    
def normalizar_minmax_0_1(serie):
    minimo = float(serie.min())   
    maximo = float(serie.max())   
    rango = maximo - minimo       # denominador de la normalización

    if rango == 0.0:
        return pd.Series(0.5, index=serie.index)

    # Escala lineal a [0, 1]
    return (serie - minimo) / rango


    # Combina componentes ya normalizados [0, 1] en un puntaje compuesto,
    # calculando un promedio ponderado fila a fila.
    # `componentes` es un dict con Series alineadas por índice
    # `pesos` es un dict con ponderaciones para esas mismas claves.
    
def combinar_componentes_normalizados(componentes, pesos, index):

    # Iniciamos el acumulador en cero para cada fila
    puntaje = pd.Series(0.0, index=index)
    suma_pesos = 0.0

    # Recorremos los componentes y, si tienen peso, sumamos su aporte
    for clave in componentes.keys():
        if clave in pesos:
            puntaje = puntaje + pesos[clave] * componentes[clave]
            suma_pesos += pesos[clave]

    # promedio ponderado
    puntaje = puntaje / suma_pesos
    return puntaje

def categorizar_por_percentil(porcentaje):
    if porcentaje <= 20:
        return "Bajo"
    elif porcentaje <= 40:
        return "Medio bajo"
    elif porcentaje <= 60:
        return "Medio"
    elif porcentaje <= 80:
        return "Medio alto"
    else:
        return "Alto"


def main():
    df = pd.read_csv(ENCUESTA_LIMPIA)
    config = cargar_configuracion(CONFIG_PUNTAJE)

    # Puntajes crudos desde mapeos  
    puntaje_barrio_crudo = puntuar_serie_desde_mapeo(df[COLUMNA_BARRIO],     config["puntajes_barrio"])
    puntaje_educacion_crudo = puntuar_serie_desde_mapeo(df[COLUMNA_EDUCACION], config["puntajes_educacion"])
    puntaje_trabajo_crudo = puntuar_serie_desde_mapeo(df[COLUMNA_TRABAJO],   config["puntajes_trabajo"])
    puntaje_ocupacion_crudo = puntuar_serie_desde_mapeo(df[COLUMNA_OCUPACION], config["puntajes_ocupacion"])

    # Normalización 0-1 por componente
    comp = {
        "barrio":    normalizar_minmax_0_1(puntaje_barrio_crudo),
        "educacion": normalizar_minmax_0_1(puntaje_educacion_crudo),
        "trabajo":   normalizar_minmax_0_1(puntaje_trabajo_crudo),
        "ocupacion": normalizar_minmax_0_1(puntaje_ocupacion_crudo),
    }

    # Pesos (del JSON, sin fallback)
    pesos = config["pesos"]

    # Puntaje compuesto 0-1 y percentil 0-100
    puntaje_compuesto_0_1 = combinar_componentes_normalizados(comp, pesos, index=df.index)
    percentil = puntaje_compuesto_0_1.rank(method="average", pct=True) * 100.0
    df[COLUMNA_PERCENTIL_NSE] = pd.to_numeric(round(percentil))

    # Categoría NSE y percentil
    df[COLUMNA_SALIDA_NSE] = percentil.map(categorizar_por_percentil)


    # Guardar
    df.to_csv(ENCUESTA_CON_NSE, index=False)

if __name__ == "__main__":
    main()
