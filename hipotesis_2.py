# hipotesis_2.py
# Hipótesis 2: NSE percibido bajo → mayor probabilidad de exposición

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===================== Configuración básica ===================== #
RUTA_CSV = "Encuesta_limpia.csv"

# Nombres de columnas (ajusta si en tu dataset se llaman distinto)
COL_EXPOSICION = "8 - ¿Conoce o recuerda algún caso de procedimientos policiales inadecuados y/o violentos?"  # "Si"/"No"
COL_NSE_CAT    = "nivel socioeconómico"     # "Bajo", "Medio bajo", "Medio", "Medio alto", "Alto"
COL_NSE_SCORE  = "percentil NSE"            # numérico (0–100, por ejemplo)

# Orden lógico del NSE para ejes/tablas
ORDEN_NSE = ["Bajo", "Medio bajo", "Medio", "Medio alto", "Alto"]

sns.set_theme(style="whitegrid")


def _columna_expuesto_booleano(df: pd.DataFrame) -> pd.Series:
    """
    Devuelve una serie booleana True/False a partir de la P8:
    'Si' => True (Expuesto/a), 'No' => False. Ignora otros valores.
    """
    s = df[COL_EXPOSICION].astype(str).str.strip().str.lower()
    mapa = {"si": True, "no": False}
    return s.map(mapa)


# ============= Gráfico 1: % de “Expuesto/a” por nivel socioeconómico ============= #
def grafico_divergente_si_no_por_nse(df: pd.DataFrame):
    """Barras 100% divergentes: composición 'Si' vs 'No' por NSE."""
    resp = df[COL_EXPOSICION].astype(str).str.strip().str.lower()
    datos = pd.DataFrame({COL_EXPOSICION: resp, COL_NSE_CAT: df[COL_NSE_CAT]}).dropna()
    datos[COL_NSE_CAT] = pd.Categorical(datos[COL_NSE_CAT], categories=ORDEN_NSE, ordered=True)

    # tabla % por NSE
    ct = (pd.crosstab(datos[COL_NSE_CAT], datos[COL_EXPOSICION], normalize="index") * 100).fillna(0)
    # asegurar columnas
    for col in ["si", "no"]:
        if col not in ct.columns: ct[col] = 0.0
    ct = ct[["no", "si"]]

    izq = -ct["no"]
    der = ct["si"]

    fig, ax = plt.subplots()
    ax.bar(ct.index, izq, label="No")
    ax.bar(ct.index, der, bottom=0, label="Sí")

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(
        title="Composición por NSE — 'Sí' vs 'No'",
        xlabel="Nivel socioeconómico (Figura 5)", ylabel="Porcentaje",
        ylim=(-100, 100),
    )
    ax.legend()
    return ax



# ===== Gráfico 2: tendencia de exposición por cuantiles del puntaje_nse ===== #
def grafico_box_puntaje_nse_por_exposicion(df: pd.DataFrame):
    """
    Boxplot de 'percentil NSE' por exposición con mediana etiquetada.
    - Grupos: Expuesto/a vs No expuesto/a (P8: Si/No → mapeo).
    - Eje Y: percentil NSE (0–100 por defecto).
    """
    # Mapear P8 a etiquetas claras
    serie_exp = (
        df[COL_EXPOSICION].astype(str).str.strip().str.lower()
        .map({"si": "Expuesto/a", "no": "No expuesto/a"})
    )

    datos = pd.DataFrame({
        "grupo_exposicion": serie_exp,
        COL_NSE_SCORE: pd.to_numeric(df[COL_NSE_SCORE], errors="coerce"),
    }).dropna()

    orden = ["Expuesto/a", "No expuesto/a"]
    datos = datos[datos["grupo_exposicion"].isin(orden)]

    # Boxplot + puntos (jitter)
    ax = sns.boxplot(
        data=datos,
        x="grupo_exposicion", y=COL_NSE_SCORE,
        order=orden, width=0.5
    )
    sns.stripplot(
        data=datos,
        x="grupo_exposicion", y=COL_NSE_SCORE,
        order=orden, color="k", alpha=0.35, size=3, jitter=True
    )

    # Medianas por grupo: marcador y etiqueta
    medianas = datos.groupby("grupo_exposicion")[COL_NSE_SCORE].median()
    for i, cat in enumerate(orden):
        if cat in medianas.index:
            y = float(medianas.loc[cat])
            ax.scatter(i, y, marker="D", s=46, color="black", zorder=5)
            ax.annotate(f"{y:.1f}", (i, y), xytext=(0, 6),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, color="black", weight="bold")

    ax.set(
        title="Distribución del percentil NSE por exposición",
        xlabel="Exposición a casos (Figura 6)",
        ylabel="Percentil NSE",
        ylim=(0, 100),  # ajustá si tu escala no es 0–100
    )
    return ax



# =============================== Main =============================== #
def main():
    df = pd.read_csv(RUTA_CSV)

    # Gráfico 1
    grafico_divergente_si_no_por_nse(df)
    plt.show()

    # Gráfico 2
    grafico_box_puntaje_nse_por_exposicion(df)
    plt.show()


if __name__ == "__main__":
    main()
