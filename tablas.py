# tabla_1.py
import pandas as pd

# --- Configuración básica ---
ruta_csv = "Encuesta.csv"
columna_genero = "1- Género"
columna_edad = "2- Edad (años):"

def clasificar_genero(valor):
    # Devuelve 'Varón', 'Mujer' o 'Otro'
    if valor == "Varon":
        return "Varón"
    if valor == "Mujer":
        return "Mujer"
    return "Otro"

def clasificar_edad(edad):
    # Devuelve '1 a 30', '31 a 60' o '61 en adelante'
    if 1 <= edad <= 30:
        return "1 a 30"
    if 31 <= edad <= 60:
        return "31 a 60"
    return "61 en adelante"

def tabla_totales_y_porcentajes(serie_categorizada, orden, nombre_variable):
    conteos = serie_categorizada.value_counts().reindex(orden).fillna(0).astype(int)
    porcentajes = (conteos / conteos.sum() * 100).round(2)
    porcentajes_str = porcentajes.map(lambda v: f"{v:.2f}%")  

    tabla = pd.DataFrame({
        nombre_variable: conteos.index,
        "Total": conteos.values,
        "Porcentaje": porcentajes_str.values
    })
    return tabla.set_index(nombre_variable)

def main():
    # Leer CSV (se asume codificación utf-8 y datos válidos)
    df = pd.read_csv(ruta_csv)

    # Categorización de variables
    genero_cat = df[columna_genero].apply(clasificar_genero)
    edad_cat = df[columna_edad].apply(clasificar_edad)

    # Tablas
    orden_genero = ["Varón", "Mujer", "Otro"]
    orden_edad = ["1 a 30", "31 a 60", "61 en adelante"]

    tabla_genero = tabla_totales_y_porcentajes(genero_cat, orden_genero, "Género")
    tabla_edad = tabla_totales_y_porcentajes(edad_cat, orden_edad, "Edad")

    # Mostrar resultados
    print("\n=== Tabla por Género ===")
    print(tabla_genero)

    print("\n=== Tabla por Edad ===")
    print(tabla_edad)

    tabla_genero.to_html("tabla_genero.html", index=True)
    tabla_edad.to_html("tabla_edad.html", index=True)


if __name__ == "__main__":
    main()
