"""
claves_busqueda.py

Clasificación de respuestas libres de residencia hacia etiquetas canónicas.
- GBA: por PARTIDO (como antes).
- CABA: por COMUNA (1 a 15). Se mantiene 'caba' como alias general cuando no se especifica barrio.

Uso con RapidFuzz:
    from rapidfuzz import process, fuzz
    from claves_busqueda import CLAVES_NORMALIZADAS, A_CANONICO
    match = process.extractOne(texto_norm, CLAVES_NORMALIZADAS, scorer=fuzz.token_set_ratio, score_cutoff=89)
    if match: etiqueta = A_CANONICO[match[0]]  # "comuna X" / partido / "caba"

La normalización (lower + sin tildes) debe hacerse ANTES, fuera de este módulo.
"""

from __future__ import annotations
from typing import Dict, List
import unicodedata


def _norm(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace(".", " ")
    s = " ".join(s.split())
    return s

# ============================================================================================
#                               DEFINICIÓN CANÓNICA
# ============================================================================================

CANONICOS: Dict[str, List[str]] = {
    # --------------------------- CABA ---------------------------
    "caba": [
        # Aliases generales (catch‑all sin barrio)
        "caba", "capital federal", "ciudad autonoma de buenos aires", "ciudad autonoma",
        "ciudad autonoma de bs as", "cap fed", "capital", "c a b a", "c.a.b.a.",
    ],

    # --- CABA por COMUNAS ---
    "comuna 1": [
        "retiro", "san nicolas", "san nícolas", "puerto madero", "san telmo",
        "monserrat", "constitucion",
    ],
    "comuna 2": [
        "recoleta",
    ],
    "comuna 3": [
        "balvanera", "san cristobal",
    ],
    "comuna 4": [
        "la boca", "barracas", "parque patricios", "nueva pompeya",
    ],
    "comuna 5": [
        "almagro", "boedo",
    ],
    "comuna 6": [
        "caballito",
    ],
    "comuna 7": [
        "flores", "parque chacabuco",
    ],
    "comuna 8": [
        "villa soldati", "villa riachuelo", "villa lugano",
    ],
    "comuna 9": [
        "liniers", "mataderos", "parque avellaneda",
    ],
    "comuna 10": [
        "villa real", "monte castro", "versalles", "floresta", "velez sarsfield", "villa luro",
    ],
    "comuna 11": [
        "villa general mitre", "villa devoto", "villa del parque", "villa santa rita",
    ],
    "comuna 12": [
        "coghlan", "saavedra", "villa pueyrredon", "villa pueyrredón", "villa urquiza",
    ],
    "comuna 13": [
        "nunez", "nuñez", "belgrano", "colegiales",
    ],
    "comuna 14": [
        "palermo",
    ],
    "comuna 15": [
        "chacarita", "villa crespo", "la paternal", "agronomia", "parque chas", "villa ortuzar",
        "agronomía", "villa ortúzar",
    ],

    # ------------------------ GBA SUR / OESTE / NORTE ------------------------
    # Avellaneda
    "avellaneda": [
        "avellaneda", "sarandi", "wilde", "gerli", "dock sud", "pineiro", "piñeiro",
        "villa dominico", "domínico", "dominico",
    ],

    # Lanus
    "lanus": [
        "lanus", "lanus oeste", "lanus este", "valentin alsina", "remedios de escalada",
        "gerli", "monte chingolo",
    ],

    # Lomas de Zamora
    "lomas de zamora": [
        "lomas de zamora", "banfield", "temperley", "llavallol", "turdera",
        "san jose", "villa fiorito",
    ],

    # Quilmes
    "quilmes": [
        "quilmes", "bernal", "bernal oeste", "ezpeleta", "san francisco solano",
        "don bosco", "bernál",
    ],

    # Berazategui
    "berazategui": [
        "berazategui", "ranelagh", "platanos", "juan maria gutierrez", "hudson",
        "villa espanola", "villa española",
    ],

    # Florencio Varela
    "florencio varela": [
        "florencio varela", "bosques", "zeballos", "ingeniero allan", "villa vane",
        "la capilla",
    ],

    # Almirante Brown
    "almirante brown": [
        "almirante brown", "adrogue", "burzaco", "glew", "claypole", "jose marmol",
        "jose marmol", "ministro rivadavia", "longchamps", "rafael calzada",
        "malvinas argentinas (brown)", "san jose (brown)",
    ],

    # La Matanza
    "la matanza": [
        "la matanza", "san justo", "ramos mejia", "lomas del mirador", "la tablada",
        "villa luzuriaga", "isidro casanova", "gregorio de laferrere", "laferrere",
        "gonzalez catan", "virrey del pino", "tapiales", "ciudad madero", "20 de junio",
    ],

    # Moron
    "moron": [
        "moron", "haedo", "castelar", "el palomar",
    ],

    # Ituzaingo
    "ituzaingo": [
        "ituzaingo", "villa udaondo", "parque leloir",
    ],

    # Hurlingham
    "hurlingham": [
        "hurlingham", "villa tesei", "william morris",
    ],

    # Tres de Febrero
    "tres de febrero": [
        "tres de febrero", "saenz pena", "santos lugares", "caseros", "villa bosch",
        "jose ingenieros", "ciudadela", "martin coronado", "once de septiembre",
        "churruca",
    ],

    # San Martin
    "san martin": [
        "san martin", "san andres", "villa ballester", "jose leon suarez",
        "billinghurst", "villa lynch", "malaver", "villa maipu",
    ],

    # San Miguel
    "san miguel": [
        "san miguel", "muniz", "muñiz", "bella vista", "campo de mayo",
    ],

    # Malvinas Argentinas
    "malvinas argentinas": [
        "malvinas argentinas", "los polvorines", "grand bourg", "tortuguitas",
        "ingeniero pablo nogues", "pablo nogues", "villa de mayo",
    ],

    # Vicente Lopez
    "vicente lopez": [
        "vicente lopez", "olivos", "la lucila", "florida", "munro", "carapachay",
        "villa adelina",
    ],

    # San Isidro
    "san isidro": [
        "san isidro", "martinez", "beccar", "acassuso", "boulogne",
        "villa adelina (san isidro)",
    ],

    # San Fernando
    "san fernando": [
        "san fernando", "victoria", "virreyes", "islas del delta",
    ],

    # Tigre
    "tigre": [
        "tigre", "general pacheco", "pacheco", "don torcuato", "rincon de milberg",
        "benavidez", "el talar", "troncos del talar", "nordelta", "don torcuato 1ra",
        "don torcuato 2da",
    ],

    # Merlo
    "merlo": [
        "merlo", "san antonio de padua", "padua", "parque san martin", "mariano acosta",
    ],

    # Moreno
    "moreno": [
        "moreno", "trujui", "trujiui", "francisco alvarez", "la reja",
        "paso del rey", "cuartel v",
    ],

    # Jose C Paz
    "jose c paz": [
        "jose c paz", "jose c. paz", "altos de jose c paz", "nueva granada",
    ],

    # Ezeiza
    "ezeiza": [
        "ezeiza", "canning", "la union", "carlos spegazzini", "spegazzini",
    ],

    # Esteban Echeverria
    "esteban echeverria": [
        "esteban echeverria", "monte grande", "9 de abril", "nueve de abril",
        "el jaguel", "luis guillon",
    ],
}

# ============================================================================================
#                    Construcción de estructuras para matching
# ============================================================================================

CLAVES_NORMALIZADAS: List[str] = []
A_CANONICO: Dict[str, str] = {}

for canon, variantes in CANONICOS.items():
    for v in variantes:
        nv = _norm(v)
        if nv not in A_CANONICO:
            CLAVES_NORMALIZADAS.append(nv)
            A_CANONICO[nv] = canon

__all__ = [
    "CANONICOS",
    "CLAVES_NORMALIZADAS",
    "A_CANONICO",
]
