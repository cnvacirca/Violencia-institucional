# main.py
import tablas
import limpieza
import nse
import hipotesis_1 as h1
import hipotesis_2 as h2


def main():
    print("▶ Tablas…")
    tablas.main()

    print("▶ Limpieza…")
    limpieza.main()

    print("▶ Nivel Socioeconómico…")
    nse.main()

    print("▶ Hipótesis 1…")
    h1.main()

    print("▶ Hipótesis 2…")
    h2.main()

    print("✅ Listo.")

if __name__ == "__main__":
    main()
