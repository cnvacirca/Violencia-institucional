# limpieza.py
import pandas as pd
import unicodedata
from rapidfuzz import process, fuzz
from claves_busqueda import CLAVES_NORMALIZADAS, A_CANONICO


def eliminar_columas_no_usadas(df: pd.DataFrame) -> pd.DataFrame:
    columnas_a_eliminar = [
        "Marca temporal",
        df.columns[df.columns.str.startswith("1")][0],
        df.columns[df.columns.str.startswith("2")][0],
        df.columns[df.columns.str.startswith("9")][0],
        df.columns[df.columns.str.startswith("10")][0],
        df.columns[df.columns.str.startswith("11")][0],
        df.columns[df.columns.str.startswith("12")][0],
        df.columns[df.columns.str.startswith("13")][0],
        df.columns[df.columns.str.startswith("14")][0],
        df.columns[df.columns.str.startswith("17")][0],
        df.columns[df.columns.str.startswith("18")][0],
        df.columns[df.columns.str.startswith("19")][0],
    ]
    return df.drop(columns=columnas_a_eliminar)


def a_minusculas(s):
    if pd.isna(s):
        return s
    return str(s).lower()


def quitar_tildes(s):
    if pd.isna(s):
        return s
    t = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in t if not unicodedata.combining(c))


def normalizar_basico(s):
    return quitar_tildes(a_minusculas(s))


# --- NUEVO: detección por fuzzy matching → etiqueta canónica de PARTIDO ---

def detectar_partido_fuzzy(texto_norm, cutoff: int = 80) -> str:
    """
    Recibe texto ya normalizado (lower + sin tildes) y devuelve
    el PARTIDO canónico (incluye 'caba' como partido) o 'otro'.
    """
    if pd.isna(texto_norm) or str(texto_norm).strip() == "":
        return "otro"
    t = str(texto_norm).strip()

    # Permite palabras extra y reorden (ideal para entradas libres)
    match = process.extractOne(
        t,
        CLAVES_NORMALIZADAS,
        scorer=fuzz.token_set_ratio,
        score_cutoff=cutoff,
    )
    if not match:
        return "otro"

    variante = match[0]
    return A_CANONICO.get(variante, "otro")


def normalizar_p6(valor):
    # De "Jubilado/a -> pase a pregunta 8" deja "Jubilado/a"
    return str(valor).split("->")[0].strip()


def corregir_p7(df, col_condicion_laboral, col_ocupacion):
    # Normaliza condicion_laboral y fuerza ocupacion = "No corresponde" cuando P6 != "Ocupado"
    df[col_condicion_laboral] = df[col_condicion_laboral].apply(normalizar_p6)
    mask = df[col_condicion_laboral] != "Ocupado"
    df.loc[mask, col_ocupacion] = "No corresponde"
    return df


def normalizar_p8(valor):
    # De "Si -> pase a pregunta 9" deja "Si"; idem para "No -> ..."
    base = str(valor).split("->")[0].strip()
    # homogeneiza "sí"/"si" y mayúsculas
    base = quitar_tildes(base).lower()
    if base.startswith("si"):
        return "Si"
    if base.startswith("no"):
        return "No"
    return str(valor).strip()


def normalizar_p20(valor):
    """Mapea la respuesta de la P20 a valores cortos: 'a veces' / 'siempre' / 'nunca' / 'otro'."""
    if pd.isna(valor):
        return "otro"
    s = quitar_tildes(str(valor)).lower().strip()
    # Buscamos las frases clave independientemente de prefijos/sufijos
    if "siempre" in s and "impune" in s:
        return "siempre"
    if ("a veces" in s or "aveces" in s) and "impune" in s:
        return "a veces"
    if "nunca" in s and "impune" in s:
        return "nunca"
    return "otro"


def main():
    # Lee el CSV original
    df = pd.read_csv("Encuesta.csv")

    # Limpia columnas que no usaremos para las hipótesis
    df = eliminar_columas_no_usadas(df)

    # Nombres de columnas a sobrescribir (prefijos según tu formulario)
    col_residencia = df.columns[df.columns.str.startswith("3")][0]
    col_barrio = df.columns[df.columns.str.startswith("4")][0]
    col_condicion_laboral = df.columns[df.columns.str.startswith("6")][0]
    col_ocupacion = df.columns[df.columns.str.startswith("7")][0]
    col_conoce_casos = df.columns[df.columns.str.startswith("8")][0]

    # Normaliza y clasifica residencia/barrio a PARTIDO canónico (o 'otro')
    df[col_residencia] = df[col_residencia].apply(normalizar_basico).apply(detectar_partido_fuzzy)
    df[col_barrio] = df[col_barrio].apply(normalizar_basico).apply(detectar_partido_fuzzy)

    # Regla de salto: solo "Ocupado" puede tener P7; el resto "No corresponde"
    df = corregir_p7(df, col_condicion_laboral, col_ocupacion)

    # Corregir la pregunta 8
    df[col_conoce_casos] = df[col_conoce_casos].apply(normalizar_p8)

    # Renombrar y normalizar la P20 (impunidad en el sistema judicial)
    col_p20_old = df.columns[df.columns.str.startswith("20")][0]
    col_p20_new = "20 - ¿Considera que los casos de abuso y violencia policial quedan impunes en el sistema judicial?"
    df.rename(columns={col_p20_old: col_p20_new}, inplace=True)
    df[col_p20_new] = df[col_p20_new].apply(normalizar_p20)

    # Guarda el CSV resultante
    df.to_csv("Encuesta_limpia.csv", index=False)



if __name__ == "__main__":
    main()