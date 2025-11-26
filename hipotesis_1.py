# hipotesis_1.py

# Hipótesis 1: Quienes han estado expuestos (testigo o conocen) 
# perciben la violencia institucional como un problema "alto/muy alto" más que quienes no.


# =============================== Imports =============================== #
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  
import numpy as np

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ============================= Configuración ============================ #
DATA_PATH = "Encuesta_limpia.csv"
sns.set_theme(style="whitegrid")


# ============================== Carga & Cols =========================== #

def cargar_datos(path: str = DATA_PATH) -> pd.DataFrame:
    """Lee el CSV de trabajo y devuelve un DataFrame."""
    return pd.read_csv(path)


def detectar_columnas(df: pd.DataFrame) -> dict:
    """
    Detecta columnas por prefijo textual (robusto a cambios de redacción).
    Retorna un dict con claves: 'p8', 'p15', 'p16', 'p20'.
    """
    def tomar_col(prefijo: str) -> str:
        cols = df.columns[df.columns.str.startswith(prefijo)]
        if len(cols) == 0:
            raise ValueError(f"No se encontró columna que empiece con '{prefijo}'")
        return cols[0]

    cols = {
        "p8": tomar_col("8"),
        "p15": tomar_col("15"),
        "p16": tomar_col("16"),
        "p20": tomar_col("20"), 
    }
    return cols


# ================================ Features ============================= #

def construir_features(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """
    Prepara el DataFrame para análisis y gráficos.

    Acciones principales:
      1) 'exposicion': deriva a partir de P8 mapeando "Si"→"Expuesto/a", "No"→"No expuesto/a"
         (cualquier otro valor queda como "Otro").
      2) Tipado numérico: convierte P15 y P16 a numérico con errors='coerce' (valores no parsables → NaN).
      3) 'relevancia_alta': bandera booleana que vale True cuando P15 ≥ 4 (alto/muy alto).
      4) P20 categórica ordenada: define el orden analítico ["siempre", "a veces", "nunca", "otro"]
         para que gráficos/tablas respeten esa secuencia (no alfabética).

    Retorna:
      DataFrame copia ('out') con columnas derivadas y tipos consistentes.

    """
    out = df.copy()

    # 1) Exposición a partir de P8
    exp_map = {"Si": "Expuesto/a", "No": "No expuesto/a"}
    out["exposicion"] = out[cols["p8"]].map(exp_map).fillna("Otro")

    # 2) Conversión a numérico para P15 y P16
    out[cols["p15"]] = pd.to_numeric(out[cols["p15"]], errors="coerce")
    out[cols["p16"]] = pd.to_numeric(out[cols["p16"]], errors="coerce")

    # 3) Bandera de relevancia alta (4 o 5)
    out["relevancia_alta"] = (out[cols["p15"]] >= 4)

    # 4) Categorización ordenada para P20
    categorias_p20 = ["siempre", "a veces", "nunca", "otro"]
    out[cols["p20"]] = pd.Categorical(out[cols["p20"]], categories=categorias_p20, ordered=True)

    return out

# ================================= Gráficos ============================ #

def grafico_proporcion_relevancia_alta_por_exposicion(df: pd.DataFrame, cols: dict):
    """
    Gráfico 1: porcentaje de personas que califican la relevancia (P15) como alta/muy alta (≥4),
    comparando grupos de exposición (P8). Eje Y en %.
    """
    # 1) Calcular proporción de 'relevancia_alta' por grupo y llevarla a porcentaje
    tabla = (
        df.dropna(subset=["relevancia_alta"])
          .groupby("exposicion")["relevancia_alta"]
          .mean() * 100.0
    ).reset_index(name="porcentaje")

    # 2) Mantener solo los grupos de interés y ordenarlos para el eje X
    orden_grupos = ["Expuesto/a", "No expuesto/a"]
    tabla = tabla[tabla["exposicion"].isin(orden_grupos)]
    tabla["exposicion"] = pd.Categorical(tabla["exposicion"], categories=orden_grupos, ordered=True)
    tabla = tabla.sort_values("exposicion")

    # 3) Dibujar barras con eje Y en porcentaje
    ax = sns.barplot(data=tabla, x="exposicion", y="porcentaje")
    ax.set(
        title="% que percibe el problema como 'alto/muy alto'",
        xlabel="Exposición a casos (Figura 1)",
        ylabel="Porcentaje",
        ylim=(0, 100),
    )
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    return ax


def grafico_distribucion_p15_por_exposicion(df: pd.DataFrame, cols: dict):
    """
    Gráfico 2: distribución de P15 (1–5) por grupo de exposición.
    Combina: violín (forma de la distribución) + puntos (casos) + mediana con intervalo.
    """
    # 1) Datos válidos y orden claro de grupos (solo los dos principales)
    orden_grupos = ["Expuesto/a", "No expuesto/a"]
    datos = df.dropna(subset=[cols["p15"], "exposicion"])
    datos = datos[datos["exposicion"].isin(orden_grupos)]

    # 2) Violín: muestra la forma de la distribución por grupo
    ax = sns.violinplot(
        data=datos,
        x="exposicion", y=cols["p15"],
        order=orden_grupos,
        cut=0, inner=None, linewidth=1,
    )

    # 3) Puntos: observaciones individuales (con jitter para evitar superposición)
    sns.stripplot(
        data=datos,
        x="exposicion", y=cols["p15"],
        order=orden_grupos,
        size=3, alpha=0.4, color="k", jitter=True,
    )

    # 4) Mediana + intervalo percentil 95% superpuestos
    #    (Usamos un estimador compatible con arrays de NumPy para evitar errores.)
    sns.pointplot(
        data=datos,
        x="exposicion", y=cols["p15"],
        order=orden_grupos,
        estimator=lambda v: float(pd.Series(v).median()),
        errorbar=("pi", 95),
        markers="D", linestyles="", color="black",
    )

    # 5) Etiquetas y límites
    ax.set(
        title="Distribución de relevancia por exposición",
        xlabel="Exposición a casos (Figura 2)",
        ylabel="Relevancia (1 = menor, 5 = mayor)",
        ylim=(1, 5),
    )
    ax.set_yticks([1, 2, 3, 4, 5])

    return ax


def grafico_donut_concentrico_p15_expuesto_vs_no(df: pd.DataFrame, cols: dict):
    """
    Donut concéntrico (2 anillos):
      Exterior: Expuesto/a
      Interior: No expuesto/a
    """

    # --- 1) Datos ---
    grupos = ["Expuesto/a", "No expuesto/a"]
    datos = df[df["exposicion"].isin(grupos)].dropna(subset=[cols["p15"], "exposicion"])

    p15_discreta = datos[cols["p15"]].round().clip(1, 5).astype(int)
    tabla = pd.crosstab(index=datos["exposicion"], columns=p15_discreta, normalize="index") * 100.0
    categorias = [1, 2, 3, 4, 5]
    for c in categorias:
        if c not in tabla.columns:
            tabla[c] = 0.0
    tabla = tabla[categorias].reindex(grupos).astype(float)

    # --- 2) Figura y estilo ---
    fig, ax = plt.subplots(figsize=(6.6, 6.2))  # un poco más alta/larga
    fig.subplots_adjust(top=0.86, bottom=0.20, left=0.08, right=0.93)  # reserva para título y leyendas
    colores = sns.color_palette("Blues", n_colors=5)

    radio_exterior, radio_interior = 1.00, 0.66
    ancho_exterior, ancho_interior = 0.30, 0.30

    def formato_pct(pct: float) -> str:
        return f"{pct:.0f}%" if pct >= 3 else ""

    # --- 3) Anillos ---
    ax.pie(
        tabla.loc["Expuesto/a"].to_numpy(dtype=float),
        labels=None, autopct=formato_pct,
        startangle=90, counterclock=False,
        colors=colores, radius=radio_exterior,
        wedgeprops=dict(width=ancho_exterior, edgecolor="white"),
        pctdistance=0.82,
    )
    ax.pie(
        tabla.loc["No expuesto/a"].to_numpy(dtype=float),
        labels=None, autopct=formato_pct,
        startangle=90, counterclock=False,
        colors=colores, radius=radio_interior,
        wedgeprops=dict(width=ancho_interior, edgecolor="white"),
        pctdistance=0.78,
    )
    ax.axis("equal")

    # --- 4) Leyendas ---
    # Categorías (colores) 
    leyenda_categorias = [
        Patch(facecolor=colores[i], edgecolor="white", label=f"Relevancia={c}")
        for i, c in enumerate(categorias)
    ]
    fig.legend(
        handles=leyenda_categorias, title="Relevancia (1–5)",
        loc="lower center", bbox_to_anchor=(0.50, 0.06),
        ncol=5, frameon=True
    )

    # Anillos (grupos) — arriba derecha, **con recuadro**
    leyenda_anillos = [
        Line2D([], [], linestyle="none", label="Exterior: Expuesto/a"),
        Line2D([], [], linestyle="none", label="Interior: No expuesto/a"),
    ]
    fig.legend(
        handles=leyenda_anillos, title="Anillos (grupos)",
        loc="upper right", bbox_to_anchor=(0.93, 0.92),
        frameon=True,       
        fancybox=True,       
        framealpha=0.95,     
        edgecolor="0.6",     
        handlelength=0, handletextpad=0.3
    )


    # --- 5) Título ---
    fig.suptitle(
        "Sensación de gravedad según exposición (Figura3)",
        y=0.94, fontsize=12
    )

    return fig, ax



def grafico_likert_p20_por_exposicion_horizontal(df: pd.DataFrame, cols: dict):
    import matplotlib.pyplot as plt

    datos = df.dropna(subset=[cols["p20"], "exposicion"]).copy()
    orden = ["Expuesto/a", "No expuesto/a"]
    mapa = {"nunca": -1, "a veces": 0, "siempre": 1, "otro": 0}
    datos["p20_mapeada"] = datos[cols["p20"]].map(mapa)

    tabla = (pd.crosstab(datos["exposicion"], datos["p20_mapeada"], normalize="index") * 100.0)
    for c in (-1,0,1):
        if c not in tabla.columns: tabla[c] = 0.0
    tabla = tabla[[ -1, 0, 1 ]].reindex([g for g in orden if g in tabla.index])

    colores = {"nunca":"#6baed6","a veces":"#c6dbef","siempre":"#2171b5"}
    fig, ax = plt.subplots(figsize=(7,3.8))

    y = np.arange(len(tabla))
    izq = tabla[-1].to_numpy()
    centro = tabla[0].to_numpy()
    der = tabla[1].to_numpy()

    # tramo izquierda (nunca)
    ax.barh(y, izq, color=colores["nunca"], label="nunca")
    # tramo centro (a veces), empieza donde termina 'nunca'
    ax.barh(y, centro, left=izq, color=colores["a veces"], label="a veces")
    # tramo derecha (siempre), empieza donde termina 'centro'
    ax.barh(y, der, left=izq+centro, color=colores["siempre"], label="siempre")

    ax.set(title="Impunidad percibida (Figura 4) ",
           ylabel="Exposición a casos", xlabel="Porcentaje", xlim=(0,100))
    ax.set_xticks([0,20,40,60,80,100])
    ax.set_yticks(y, tabla.index)
    ax.legend(title="Impunidad", loc="lower right")
    return ax



# ================================== Main =============================== #

def main():
    df = cargar_datos(DATA_PATH)
    cols = detectar_columnas(df)
    df_features = construir_features(df, cols)

    # Gráfico 1
    grafico_proporcion_relevancia_alta_por_exposicion(df_features, cols)
    plt.show()


    # Gráfico 2
    grafico_distribucion_p15_por_exposicion(df_features, cols)
    plt.show()

    # Gráfico 3
    grafico_donut_concentrico_p15_expuesto_vs_no(df_features, cols)
    plt.show()


    # Gráfico 4
    grafico_likert_p20_por_exposicion_horizontal(df_features, cols)
    plt.show()





if __name__ == "__main__":
    main()
