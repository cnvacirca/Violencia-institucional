

## Estructura de archivos
.
├── Encuesta.csv                # Datos crudos (entrada)
├── tabla_1.py                  # Extrae el espacio muestral y genera un HTML por edad y género
├── limpieza.py                 # Limpieza/normalización → genera Encuesta_limpia.csv
├── Encuesta_limpia.csv         # Datoos limpios (salida de limpieza.py)
├── nse.py                      # Calcula puntaje_nse, percentil NSE y "nivel socioeconómico"
├── config_nse.json             # Config (pesos, mapeos y puntajes territoriales)
├── claves_busqueda.py          # Diccionarios y alias por partido/comuna + rapidfuzz
├── hipotesis_1.py              # Gráficos y análisis H1
├── hipotesis_2.py              # Gráficos y análisis H2 (+ mapa CABA opcional)
├── main.py                     # Orquestador: limpieza → nse → h1 → h2
└── README.md

---

## Cómo se relacionan los archivos

**Flujo de trabajo (pipeline)**

* **Tabla 1(tabla_1.py)**
    Genera una tabla para visualizar el espacio muestral de encuestados, en números absolutos y porcentuales.

* **Limpieza (limpieza.py)**

    Normaliza texto (lower/tildes), clasifica residencia/barrio → partido/comuna (alias + fuzzy).

    Estandariza P8 (Si/No), convierte P15–P16 a numérico.

    Renombra/normaliza P20 y ordena sus categorías.

    Exporta Encuesta_limpia.csv.

* **NSE (nse.py + config_nse.json)**

    Lee Encuesta_limpia.csv y config_nse.json.

    Calcula puntaje_nse (combinación ponderada) y percentil NSE.

    Clasifica nivel socioeconómico ∈ {Bajo, Medio bajo, Medio, Medio alto, Alto}.

    Sobrescribe/actualiza Encuesta_limpia.csv con estas columnas.

* **Hipótesis 1 (hipotesis_1.py)**

    Construye features analíticas (exposición, relevancia_alta, orden de P20).

    Gráficos seleccionados:

    % “alto/muy alto” (P15≥4) por exposición.

    Distribución P15 por exposición (violín/mediana).

    Donut concéntrico P15 (Expuesto/a vs No expuesto/a).

    Likert divergente P20 por exposición.

* **Hipótesis 2 (hipotesis_2.py)**

    Relación NSE percibido ↔ exposición.

    Gráficos esenciales:

    % Expuesto/a por nivel socioeconómico.

    Tendencia de exposición por cuantiles del percentil NSE.

