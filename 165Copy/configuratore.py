import streamlit as st
import pandas as pd
import numpy as np
import torch
import asyncio
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
from PIL import Image
import json
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def resource_path(*parts):
    return BASE_DIR.joinpath(*parts)

@st.cache_data
def load_tfidf_cached(descrizioni):
    tfidf_path = resource_path("tfidf_dict.npy")
    if tfidf_path.exists():
        return np.load(tfidf_path, allow_pickle=True).tolist()
    return calcola_tfidf(descrizioni)


# Corregge il problema dei loop async in Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(layout="wide")


import base64
from io import BytesIO

image_path = resource_path("Valpres.png")

if os.path.exists(image_path):
    image = Image.open(image_path)

    # Ridimensiona
    width_desired = 600
    width_percent = width_desired / float(image.size[0])
    height_proportional = int(float(image.size[1]) * width_percent)
    image = image.resize((width_desired, height_proportional), resample=Image.Resampling.LANCZOS)

    # Converti in base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    # Mostra logo centrato sopra ai bottoni
    st.markdown(
        f"""
        <div style="text-align: left; margin-left: 22%;">
            <img src="data:image/png;base64,{img_str}" width="{width_desired}">
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.error(f"⚠️ L'immagine non è stata trovata. Controlla il percorso: {os.path.abspath(image_path)}")



# Carica il dizionario di sinonimi
DICTIONARY_PATH = resource_path("valve_dictionary_english.json")

with open(DICTIONARY_PATH, "r", encoding="utf-8") as file:
    valve_dict = json.load(file)

MATERIAL_DICT_PATH = resource_path("material_dictionary.json")
with open(MATERIAL_DICT_PATH, "r", encoding="utf-8") as f:
    material_dict = json.load(f)

FLANGE_DICT_PATH = resource_path("flange_dictionary.json")
with open(FLANGE_DICT_PATH, "r", encoding="utf-8") as file:
    flange_dict = json.load(file)



st.markdown(
    """
    <style>
        /* Aggiunge spazio tra le colonne */
        .block-container {
            padding-left: 40px;
            padding-right: 40px;
        }
        .css-1kyxreq {
            margin-right: 20px;  /* Aggiunge spazio tra i blocchi */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Percorso del file Excel
FILE_PATH = resource_path("Lavoro per AI.xlsx")

@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer, util
    globals()["util"] = util
    return SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

@st.cache_data
def carica_excel(path):
    return pd.read_excel(path)

@st.cache_data
def calcola_tfidf(descrizioni):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descrizioni)
    feature_names = vectorizer.get_feature_names_out()
    return [
        {word: tfidf_matrix[i, idx] for idx, word in enumerate(feature_names)}
        for i in range(tfidf_matrix.shape[0])
    ]


@st.cache_data
def carica_embedding(path, descrizioni, _model):
    if os.path.exists(path):
        embeddings = np.load(path, allow_pickle=True).tolist()
        if len(embeddings) != len(descrizioni):
            st.warning("⚠️ Embeddings salvati non corrispondono alle descrizioni. Li ricalcolo...")
            embeddings = _model.encode(descrizioni, show_progress_bar=True)
            np.save(path, np.array(embeddings, dtype=np.float32))
            embeddings = embeddings.tolist()
    else:
        embeddings = _model.encode(descrizioni, show_progress_bar=True)
        np.save(path, np.array(embeddings, dtype=np.float32))
        embeddings = embeddings.tolist()
    
    return embeddings

def calcola_prezzo_usd(prezzo_eur):
    try:
        if pd.isna(prezzo_eur):
            return "On request"
        return float(prezzo_eur) * 1.10
    except (ValueError, TypeError):
        return "On request"




def main():
    
      
    col_spacer1, col_btn1, col_btn2, col_btn3, col_spacer2 = st.columns([1, 1, 1, 1, 1])

    
    with col_btn1:
        if st.button("ChatBot"):
            st.session_state.page = "descrizione"
    
    with col_btn2:
        if st.button("Search with filters"):
            st.session_state.page = "filtri"

    with col_btn3:
        if st.button("🛒 Cart"):
            st.session_state.page = "carrello"

    if "page" not in st.session_state:
        st.session_state.page = "descrizione"  # Imposta la pagina predefinita

# Inizializza le variabili del carrello e dei prodotti selezionati se non esistono
    if "carrello" not in st.session_state:
        st.session_state.carrello = []

    if "prodotti_selezionati" not in st.session_state:
        st.session_state.prodotti_selezionati = []

    if "prodotti_selezionati_filtri" not in st.session_state:
        st.session_state.prodotti_selezionati_filtri = []


# Navigazione tra le pagine
    if st.session_state.page == "descrizione":
        ricerca_per_descrizione()
    elif st.session_state.page == "filtri":
        ricerca_per_filtri()
    elif st.session_state.page == "carrello":
        visualizza_carrello()
    
    
# Reset delle chiavi SOLO quando si cambia pagina
    # Inizializza lo stato della pagina precedente
    if "last_page" not in st.session_state:
        st.session_state.last_page = None

# Reset della query solo se si cambia pagina
    if st.session_state.page != st.session_state.last_page:
        if st.session_state.page == "descrizione":
            st.session_state.query_descrizione = ""

    st.session_state.last_page = st.session_state.page  # Aggiorna la pagina attuale

def estrai_condizioni_operativa(query):
    query = query.lower()

    temperatura_c = None
    pressione_psi = None

    # Estrai °F → converti in °C
    temp_match_f = re.search(r'(-?\d+)[ ]?(°f|f|fahrenheit)', query)
    if temp_match_f:
        temp_f = int(temp_match_f.group(1))
        temperatura_c = round((temp_f - 32) * 5/9, 2)

    # Estrai °C
    temp_match_c = re.search(r'(-?\d+)[ ]?(°c|c|celsius)', query)
    if temp_match_c:
        temperatura_c = int(temp_match_c.group(1))

    # Estrai PSI
    press_match_psi = re.search(r'(\d+)[ ]?(psi)', query)
    if press_match_psi:
        pressione_psi = int(press_match_psi.group(1))

    # Estrai BAR → converti in PSI
    press_match_bar = re.search(r'(\d+)[ ]?(bar|bars)', query)
    if press_match_bar:
        bar = int(press_match_bar.group(1))
        pressione_psi = round(bar * 14.5038, 2)

    return temperatura_c, pressione_psi

def estrai_dimensione(query):
    query = query.lower()
    dn = None
    pollici = None

    # Cerca DN (es. DN50 o DN 50)
    match_dn = re.search(r'\bdn[ -]?(\d+)\b', query)
    if match_dn:
        try:
            dn = int(match_dn.group(1))
        except:
            pass

    # Caso: combinazione tipo 1"1/4
    match_mix = re.search(r'(\d+)"\s*(\d+)/(\d+)', query)
    if match_mix:
        try:
            intero = int(match_mix.group(1))
            num = int(match_mix.group(2))
            den = int(match_mix.group(3))
            pollici = round(intero + num / den, 2)
            return dn, pollici
        except:
            pass

    # Caso: frazione pura tipo 3/4"
    match_frac = re.search(r'(\d+)/(\d+)"', query)
    if match_frac:
        try:
            num = int(match_frac.group(1))
            den = int(match_frac.group(2))
            pollici = round(num / den, 2)
            return dn, pollici
        except:
            pass

    # Caso: intero o decimale seguito da inch o virgolette
    match_pollici = re.search(r'(\d+(?:[.,]\d+)?)[ ]?(inches|inch|in\.|\bin\b|")', query)
    if match_pollici:
        try:
            pollici = float(match_pollici.group(1).replace(',', '.'))
        except:
            pass

    return dn, pollici




def converti_pollici_complessi(val):
    val = str(val).strip().replace('"', '').replace(' ', '')

    # Caso: 1"1/4 → 1.25
    match = re.match(r'(\d+)"?(\d+)/(\d+)', val)
    if match:
        intero = int(match.group(1))
        num = int(match.group(2))
        den = int(match.group(3))
        return round(intero + num / den, 3)

    # Caso: solo frazione → 3/4"
    match_frac = re.match(r'(\d+)/(\d+)', val)
    if match_frac:
        return round(int(match_frac.group(1)) / int(match_frac.group(2)), 3)

    # Caso: solo intero → 2", 4", ecc.
    try:
        return float(val)
    except:
        return None


def ricerca_per_descrizione():
    st.title("ChatBot")
    
    
    
    
    # Inizializza lo stato della query SE NON esiste
    if "query_descrizione" not in st.session_state:
        st.session_state.query_descrizione = ""

    # Usa una chiave FISSA per il text_input senza cambiare dinamicamente
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Enter the valve type, materials, and any customizations to find the most suitable product:",
            key="query_descrizione_input"
        )
    # Aggiorna il valore nel session state
    st.session_state.query_descrizione = query


    # Caricare il file Excel
    # Caricamento del file Excel
    df = carica_excel(FILE_PATH)

    # Normalizzazione colonne
    df["descrizione_lower"] = df["descrizione"].astype(str).str.lower().str.strip()

    df["Body material 2"] = df["Body material 2"].astype(str).str.lower().str.strip()

    # Caricamento modello
    model = load_model()

    # ✅ Caricamento embeddings sincronizzati (evita errore di dimensione)
    embeddings = carica_embedding(resource_path("embeddings.npy"), df["descrizione_lower"].tolist(), model)
    df["embedding"] = embeddings

    # ✅ Caricamento TF-IDF (assumendo tu abbia già salvato tfidf_dict.npy correttamente)
    df["TF-IDF"] = load_tfidf_cached(df["descrizione_lower"].tolist())

    # Conversione prezzo in numerico
    df["Gross Price"] = pd.to_numeric(df["Gross Price"], errors='coerce')

    # Layout con due colonne
    col_ricerca, spazio, col_sconti = st.columns([3, 0.3, 1.2])


    with col_ricerca:
        st.header("What are you looking for?")
    
    # Caricamento del modello NLP
    model = load_model()

    
    # Calcolo embedding se non già salvati
    df["embedding"] = carica_embedding(resource_path("embeddings.npy"), df["descrizione_lower"].tolist(), model)

    
    # Calcolo TF-IDF
    df["TF-IDF"] = load_tfidf_cached(df["descrizione_lower"].tolist())


     




    
    if query:
        query = query.lower().strip()
        query = normalizza_custom_terms(query)
        query = normalizza_materiali(query)
        query = normalizza_flangiature(query)
        query = normalizza_surface(query)
        temperatura_c, pressione_psi = estrai_condizioni_operativa(query)

        flange_rilevata = trova_flangiatura(query)

# Applica filtro sulla colonna "Flange" solo se trova la flangiatura
        if flange_rilevata:
            st.info(f"🧩 Flange detected: {flange_rilevata.upper()}, applying filter")
            df["Flange"] = df["Flange"].astype(str).str.strip().str.lower()
            df_filtrato = df[df["Flange"] == flange_rilevata.lower()]
    
            if not df_filtrato.empty:
                df = df_filtrato
            else:
                st.warning(f"⚠️ No products found with flange {flange_rilevata.upper()}, showing all matching products instead.")


        dn_rilevato, pollici_rilevati = estrai_dimensione(query)
        df = filtra_stem_sealing(query, df)

        if dn_rilevato:
            st.info(f"📏 Filtering for DN = {dn_rilevato}")
            df = df[df["DN"].astype(str).str.lower() == f"dn{dn_rilevato}"]

        elif pollici_rilevati:
            st.info(f"📏 Filtering for Inches = {pollici_rilevati}")
            try:
        # Converti la colonna a float per confronto preciso
                df["Pollici"] = df["Pollici"].apply(converti_pollici_complessi)

                df = df[np.isclose(df["Pollici"], pollici_rilevati, atol=0.1)]
            except:
                st.warning("⚠️ Errore nel confronto con dimensione in pollici.")


        if temperatura_c is not None:
            st.info(f"🌡️ Filtering for temperature range including {temperatura_c}°C")
    
    # Condizione: temperatura minima ≤ richiesta ≤ temperatura massima
            df = df[(df["Temperatura Max"] >= temperatura_c) & (df["Temperatura Min"] <= temperatura_c)]

        if pressione_psi is not None:
            st.info(f"⚙️ Filtering for working pressure ≥ {pressione_psi} PSI")
            df = df[df["PSI"] >= (pressione_psi - 0.5)]

        materiale_rilevato = trova_materiale(query)
        if materiale_rilevato:
            st.info(f"🔎 Filtering for material: {materiale_rilevato}")
            df = df[df["Body material 2"] == materiale_rilevato]
        query_embedding = model.encode(query, convert_to_tensor=True)
        parole_chiave_query = set(re.findall(r'\b\w+\b', query))

        # Filtro per trattamento superficiale dal testo query
        surface_options = ["blued", "white zinc-plated", "zinc-plated", "painted", "no"]
        surface_rilevata = next((s for s in surface_options if s in query), None)

        if surface_rilevata:
            st.info(f"🎨 Filtering for surface treatment: {surface_rilevata}")
            df["Surface"] = df["Surface"].astype(str).str.strip().str.lower()
            df = df[df["Surface"] == surface_rilevata]



    # Filtra i dati in base alla categoria riconosciuta dal dizionario
         # Filtra i dati in base alla categoria riconosciuta dal dizionario
        valve_type_detected = trova_valve_type(query)
        

        if valve_type_detected in ["Flanged", "Threaded", "Welded"]:
            gruppi_valvole = {
                "Flanged": [
                    "wafer valve",
                    "split body",
                    "split wafer",
                    "split body cast iron",
                    "split body api608"
                ],
                "Threaded": [
                    "three-way valve",
                    "three-piece valve",
                    "threaded monoblock valve",
                    "two-piece valve"
                ],
                "Welded": [
                    "three-way valve",
                    "three-piece valve",
                    "threaded monoblock valve",
                    "two-piece valve"
                ]
            }

            tipi_valvola = gruppi_valvole.get(valve_type_detected, [])
            df = df[df["Valve"].str.lower().isin([x.lower() for x in tipi_valvola])]

        elif valve_type_detected:
            df = df[df["Valve"].str.lower() == valve_type_detected.lower()]


        
        df["Match TF-IDF"] = df["TF-IDF"].apply(lambda row: sum(row.get(parola, 0) for parola in parole_chiave_query if isinstance(row, dict)))
        df["Similarità"] = [util.pytorch_cos_sim(query_embedding, emb)[0].item() for emb in df["embedding"]]
        df["Score Totale"] = df["Similarità"] * 0.5 + df["Match TF-IDF"] * 0.4
        df = df.sort_values(by="Score Totale", ascending=False)
        
        # Prendiamo i 10 migliori risultati
        migliori_risultati = df.head(10)

        if "prodotti_selezionati" not in st.session_state:
            st.session_state.prodotti_selezionati = []
        st.subheader("Top 10 Products Found:")
        
        for idx, row in migliori_risultati.iterrows():
            prodotto_id = f"{row['Material']}_{row['Series']}_{idx}"
            

            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                selezionato = st.checkbox("Select", key=f"check_{prodotto_id}")



            with col2:
                gross_price_usd = calcola_prezzo_usd(row["Gross Price"])

                

                st.write(f"🔹 **{row['Material']}** - {row['Body material']} - {row['Flangiatura']} - DN: {row['DN']}, Inch: {row['Pollici']}")
                st.markdown(f"📌 **{row['Short Description']}**")
                temp_min_f = (row["Temperatura Min"] * 9/5) + 32
                temp_max_f = (row["Temperatura Max"] * 9/5) + 32
                st.write(f"🌡️ **Temperature:** {row['Temperatura Min']}°C / {temp_min_f:.1f}°F - {row['Temperatura Max']}°C / {temp_max_f:.1f}°F | ⚙️ **Pressure:** {row['BAR']} BAR / {row['PSI']} PSI")
            
                if isinstance(gross_price_usd, float):
                    st.write(f"💰 **Gross Price (USD):** {gross_price_usd:.2f}$")
                else:
                    st.write(f"💰 **Gross Price (USD):** {gross_price_usd}")
                
                st.markdown(f"{row['descrizione'].replace(chr(10), '<br>')}", unsafe_allow_html=True)
            # Cerca il PDF nella cartella pdf_disegni/
                pdf_filename = resource_path("pdf_disegni", f"{row['Series']}.pdf")
                if os.path.exists(pdf_filename):
                    with open(pdf_filename, "rb") as pdf_file:
                        st.download_button(label="📄 Download PDF drawing", data=pdf_file, file_name=f"{row['Series']}.pdf", mime="application/pdf", key=f"pdf_{row['Series']}_{idx}")
                else:
                    st.write("❌ PDF Drawing not available.")

            if selezionato:
                prodotto = {
                    "Material": row["Material"],
                     "Short Description": row["Short Description"],
                     "DN": row["DN"],
                     "Pollici": row["Pollici"],
                     "Gross Price (USD)": calcola_prezzo_usd(row["Gross Price"]),
                     "descrizione": row["descrizione"],  # Aggiunta la descrizione completa
                     "Series": row["Series"],
                }
                # Aggiungi il prodotto alla lista solo se non è già presente
                if "prodotti_selezionati" not in st.session_state:
                    st.session_state.prodotti_selezionati = []

                if selezionato and prodotto not in st.session_state.prodotti_selezionati:
                    st.session_state.prodotti_selezionati.append(prodotto)
                elif not selezionato and prodotto in st.session_state.prodotti_selezionati:
                    st.session_state.prodotti_selezionati.remove(prodotto)

        
        
                
        st.markdown("---")

# Bottone per aggiungere i prodotti selezionati al carrello
        if st.button("🛒 Add selected items to cart", key="add_to_cart_descrizione"):

            if "carrello" not in st.session_state:
                st.session_state.carrello = []

            if len(st.session_state.prodotti_selezionati) == 0:
                st.warning("⚠️ Nessun prodotto selezionato! Seleziona almeno un prodotto prima di aggiungerlo al carrello.")
            else:
                st.session_state.carrello.extend(st.session_state.prodotti_selezionati)
                st.session_state.prodotti_selezionati = []  # Svuota la lista dopo aver aggiunto i prodotti
                st.success("✅ Add selected products to cart!")



def trova_valve_type(query):
    query = query.lower().strip()
    best_match = None
    best_level = -1

    for valve_type, details in valve_dict.items():
        synonyms = details.get("synonyms", [])
        level = details.get("level", 0)

        if any(syn.lower() in query for syn in synonyms):
            if level > best_level:
                best_match = valve_type
                best_level = level

    return best_match

def normalizza_materiali(query):
    query = query.lower()
    for materiale, details in material_dict.items():
        for sinonimo in details.get("synonyms", []):
            if sinonimo in query:
                query = query.replace(sinonimo, materiale)
    return query

def normalizza_surface(query):
    with open(resource_path("surface_dictionary.json"), "r", encoding="utf-8") as file:
        surface_dict = json.load(file)

    query = query.lower()
    for trattamento, details in surface_dict.items():
        for sinonimo in details.get("synonyms", []):
            if sinonimo in query:
                query = query.replace(sinonimo, trattamento)
    return query


def normalizza_flangiature(query):
    with open(resource_path("flange_dictionary.json"), "r", encoding="utf-8") as file:
        flange_dict = json.load(file)

    query = query.lower()
    for flangia, details in flange_dict.items():
        for sinonimo in details.get("synonyms", []):
            if sinonimo in query:
                query = query.replace(sinonimo, flangia)
    return query

def normalizza_custom_terms(query):
    with open(resource_path("custom_synonyms.json"), "r", encoding="utf-8") as file:
        custom_dict = json.load(file)

    query = query.lower()
    for termine_standard, details in custom_dict.items():
        for sinonimo in details.get("synonyms", []):
            if sinonimo in query:
                query = query.replace(sinonimo, termine_standard)
    return query

def trova_flangiatura(query):
    query = query.lower().strip()
    with open(resource_path("flange_dictionary.json"), "r", encoding="utf-8") as file:
        flange_dict = json.load(file)

    for standard, details in flange_dict.items():
        for sinonimo in details.get("synonyms", []):
            if sinonimo in query or standard in query:
                return standard  # es: "#150", "pn16", "jis16k"
    return None


def trova_materiale(query):
    query = query.lower().strip()
    best_match = None

    for materiale, details in material_dict.items():
        for sinonimo in details.get("synonyms", []):
            if sinonimo in query or materiale in query:
                best_match = materiale
                break
        if best_match:
            break

    return best_match

def filtra_stem_sealing(query, df):
    query = query.lower()
    with open(resource_path("stem_sealing_dictionary.json"), "r", encoding="utf-8") as file:
        sealing_dict = json.load(file)

    df["Stem sealing"] = df["Stem sealing"].astype(str).str.strip().str.lower()

    for sealing, values in sealing_dict.items():
        synonyms = values.get("synonyms", [])
        if any(s in query for s in synonyms):
            st.info(f"🧪 Filtering: Stem sealing = {sealing}")
            return df[df["Stem sealing"] == sealing]

    return df  # Se nessun match, torna il dataframe originale


def ricerca_per_filtri():
    st.title("Search with filters")


    model = load_model()
    
    
    # Caricare il file Excel
    df = carica_excel(FILE_PATH)
    df_slider_ref = df.copy()

    # Caricamento TF-IDF e embedding già pronti
    model = load_model()
    df["descrizione_lower"] = df["descrizione"].astype(str).str.lower().str.strip()
    df["embedding"] = carica_embedding(resource_path("embeddings.npy"), df["descrizione_lower"].tolist(), model)
    df["TF-IDF"] = load_tfidf_cached(df["descrizione_lower"].tolist())
    


    if "prodotti_selezionati_filtri" not in st.session_state:
        st.session_state.prodotti_selezionati_filtri = []

    # Normalizziamo la colonna "Flangiatura"
    df["Flangiatura"] = df["Flangiatura"].astype(str).str.strip().str.lower()
    df["Categoria"] = df["Categoria"].astype(str).str.strip().str.lower()
# Normalizziamo le nuove colonne per evitare problemi con spazi e maiuscole
    df["Stem sealing"] = df["Stem sealing"].astype(str).str.strip().str.lower()
    df["Body material"] = df["Body material"].astype(str).str.strip().str.lower()

# Convertiamo "Temperatura Min" e "Temperatura Max" in numerico, forzando gli errori a NaN
    df["Temperatura Min"] = pd.to_numeric(df["Temperatura Min"], errors='coerce')
    df["Temperatura Max"] = pd.to_numeric(df["Temperatura Max"], errors='coerce')
    
    # Calcoliamo i valori minimi e massimi PRIMA di filtrare
    # Calcola i range degli slider dalla copia originale (così non vengono influenzati dai filtri successivi)
    temp_min_min = df_slider_ref["Temperatura Min"].min()
    temp_min_max = df_slider_ref["Temperatura Min"].max()
    temp_max_min = df_slider_ref["Temperatura Max"].min()
    temp_max_max = df_slider_ref["Temperatura Max"].max()

    df_slider_ref["BAR"] = pd.to_numeric(df_slider_ref["BAR"], errors='coerce').fillna(0)
    df_slider_ref["PSI"] = df_slider_ref["BAR"] * 14.5038  # 1 BAR = 14.5038 PSI


        
    col_filtro, spazio1, col_risultati, spazio2, col_sconti = st.columns([1.2, 0.2, 3, 0.2, 1.2])


    with col_sconti:
        st.header("Certifications")

        # Pre-elaboriamo le colonne certificazione per evitare problemi
        for cert in ["ATEX", "FIRE SAFE", "Fugitive Emission Tested", "SIL3", "ADR", "DVGW"]:
            df[cert] = df[cert].astype(str).str.lower().str.strip()

        # Checkbox certificazioni
        cert_atex = st.checkbox("✅ ATEX Certified")
        cert_fire_safe = st.checkbox("🔥 Fire Safe Tested")
        cert_fugitive = st.checkbox("🧪 Fugitive Emission Tested")
        if cert_fugitive:
            df = df[df["Fugitive Emission Tested"] == "si"]

    # Prepara anche le due colonne da normalizzare
            for sub_cert in ["T.A Luft", "ISO15848"]:
                df[sub_cert] = df[sub_cert].astype(str).str.lower().str.strip()

    # Mostra i sottofiltri
            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                cert_taluft = st.checkbox("T.A Luft")
                if cert_taluft:
                    df = df[df["T.A Luft"] == "si"]

            with col_sub2:
                cert_iso15848 = st.checkbox("ISO15848")
                if cert_iso15848:
                    df = df[df["ISO15848"] == "si"]
        cert_sil3 = st.checkbox("🛡️ SIL3 Certified")
        cert_adr = st.checkbox("🚛 ADR Certified")
        cert_dvgw = st.checkbox("🛢️ DVGW Certified")

        # Applichiamo i filtri
        if cert_atex:
            df = df[df["ATEX"] == "si"]

        if cert_fire_safe:
            df = df[df["FIRE SAFE"] == "si"]

        if cert_fugitive:
            df = df[df["Fugitive Emission Tested"] == "si"]

        if cert_sil3:
            df = df[df["SIL3"] == "si"] 

        if cert_adr:
            df = df[df["ADR"] == "si"] 

        if cert_dvgw:
            df = df[df["DVGW"] == "si"]  

    
    with col_filtro:
        st.header("Filters")

        # Filtro per tipo di servizio (On/Off o Control)
        st.markdown("### ⚙️ Select the service mode")
        service_type = st.radio(
            "Choose the valve operation mode:",
            ["All", "On/Off", "Control"],
            horizontal=True,
            help="Control valves = fine regulation (Ball: control). On/Off = fully open/closed"
        )

        df["Ball"] = df["Ball"].astype(str).str.lower().str.strip()
        if service_type == "Control":
            df = df[df["Ball"] == "control"]
        elif service_type == "On/Off":
            df = df[df["Ball"] != "control"]

        
        st.markdown("""<hr style="margin: 10px 0; border: 0.5px solid #ccc;">""", unsafe_allow_html=True)
        st.write("")
        st.markdown("### ️ Temperature range")



# Opzione Celsius o Fahrenheit
        temp_unit = st.radio("Temperature unit", ["Celsius", "Fahrenheit"], index=0)
        
        # Se l'utente seleziona Fahrenheit, convertiamo i valori nel DataFrame
        if temp_unit == "Fahrenheit":
            df["Temperatura Min"] = df["Temperatura Min"] * 9/5 + 32
            df["Temperatura Max"] = df["Temperatura Max"] * 9/5 + 32

            temp_min_min = temp_min_min * 9/5 + 32
            temp_min_max = temp_min_max * 9/5 + 32
            temp_max_min = temp_max_min * 9/5 + 32
            temp_max_max = temp_max_max * 9/5 + 32

        
        
                
        # Slider per la Temperatura Minima
        # Prepara slider per Temperatura Minima
        if int(temp_min_min) == int(temp_min_max):
            temp_min_max += 1  # Evita errore slider
        temp_min = st.slider(
            f"Minimum temperature ({temp_unit})",
            min_value=int(temp_min_min),
            max_value=int(temp_min_max),
            value=int(temp_min_max)
        )
        df = df[df["Temperatura Min"].round() <= temp_min]

# Prepara slider per Temperatura Massima
        if int(temp_max_min) == int(temp_max_max):
            temp_max_max += 1  # Evita errore slider  
        temp_max = st.slider(
            f"Maximum temperature ({temp_unit})",
            min_value=int(temp_max_min),
            max_value=int(temp_max_max),
            value=int(temp_max_min)
        )
        df = df[df["Temperatura Max"].round() >= temp_max]

        
        st.markdown("### ️ Pressure rating")

        
        # Slider per Pressione (Bar o PSI)
        pressure_unit = st.radio("Pressure unit", ["BAR", "PSI"], index=0)
        
        # Pulizia NaN prima di calcolare i minimi e massimi
        if pressure_unit == "BAR":
            pressure_min = int(df_slider_ref["BAR"].min())
            pressure_max = int(df_slider_ref["BAR"].max())
        else:
            pressure_min = int(df_slider_ref["PSI"].min())
            pressure_max = int(df_slider_ref["PSI"].max())

        
        if pressure_min == pressure_max:
            pressure_max += 1
        
        pressure_value = st.slider(f"Working pressure ({pressure_unit})", min_value=pressure_min, max_value=pressure_max, value=pressure_min)
        df = df[df[pressure_unit] >= pressure_value]

        
        st.markdown("""<hr style="margin: 10px 0; border: 0.5px solid #ccc;">""", unsafe_allow_html=True)
        st.write("")
        st.markdown("### ️ End connections")

        
        # Filtro End Connections
        end_conn = st.radio("End Connections", ["Threaded", "Flanged", "Welded"], index=0)
        if end_conn == "Threaded":
            df = df[df["Flangiatura"].str.lower() == "threaded"]
            threaded_types = ["ISO 7/1", "ISO 228-1", "NPT"]
            selected_threaded = st.multiselect("Threaded Type", threaded_types, default=threaded_types)
            df = df[df["connections"].astype(str).str.upper().isin([x.upper() for x in selected_threaded])]

        elif end_conn == "Welded":
            df = df[df["Flangiatura"].str.lower() == "welded"]
            welded_types = ["BW", "SW"]
            selected_welded = st.multiselect("Welded Type", welded_types, default=welded_types)
            df = df[df["connections"].astype(str).str.upper().isin([x.upper() for x in selected_welded])]

        else:  # cioè se end_conn == "Flanged"
            df = df[df["Flangiatura"].str.lower() != "threaded"]

    # Mostra SEMPRE il filtro Standard se almeno uno è presente nel dataset completo
            if not df_slider_ref["Standard"].isna().all():
                standard_options = ["American", "European", "Japanese"]
                standard = st.radio("Standard", standard_options, index=0, key="standard_radio_flanged")

        # Applica il filtro sul df corrente
                df = df[df["Standard"].astype(str).str.lower() == standard.lower()]

                if df.empty:
                    st.warning(f"⚠️ No products found for selected certification(s) and **{standard}** standard.")

        # Filtro dinamico su 'Flange'
                if standard == "American":
                    flange_options = ["All", "#150", "#300", "#600"]
                elif standard == "Japanese":
                    flange_options = ["All", "JIS16K", "JIS5K", "JIS20K", "JIS"]
                else:
                    flange_options = ["All", "PN10", "PN16", "PN16/40", "PN40", "PN63", "PN100"]

                selected_flange = st.selectbox("Flange rating", flange_options, index=0)
                if selected_flange != "All":
                    df = df[df["Flange"].astype(str).str.upper() == selected_flange.upper()]


        st.markdown("""<hr style="margin: 10px 0; border: 0.5px solid #ccc;">""", unsafe_allow_html=True)
        st.write("")
        st.markdown("### ️ Valve style")

        
        # Filtro Tipo di Valvola
        valve_types = ["split body", "unibody", "2 pcs", "2 pcs solid bar", "3 pcs", "3 way", "wafer", "wafer split"]
        valve_selected = st.multiselect("Valve Type", valve_types, default=valve_types)
        df = df[df["Categoria"].isin(valve_selected)]

        st.markdown("""<hr style="margin: 10px 0; border: 0.5px solid #ccc;">""", unsafe_allow_html=True)
        st.write("")
        st.markdown("### ️ Materials")


        # Filtro per Stem Sealing
        stem_options = ["EPDM", "FKM", "FFKM", "HNBR", "FKM Perox", "MFQ", "V-PACK PTFE (For Chemicals)", "NBR"]
        stem_selected = st.multiselect("Stem Sealing", stem_options, default=stem_options)
        df = df[df["Stem sealing"].str.lower().isin([x.lower() for x in stem_selected])]

        st.markdown("""<hr style="margin: 10px 0; border: 0.5px solid #ccc;">""", unsafe_allow_html=True)


# Filtro per Body Material
        body_options = ["Carbon steel", "Stainless steel", "Stainless steel from solid bar",
                "Carbon steel from solid bar", "Cast iron", "Super duplex", "Brass"]
        body_selected = st.multiselect("Body Material", body_options, default=body_options)
        df = df[df["Body material"].str.lower().isin([x.lower() for x in body_selected])]

        st.markdown("""<hr style="margin: 10px 0; border: 0.5px solid #ccc;">""", unsafe_allow_html=True)
        st.write("")
        st.markdown("### ️ Surface treatments")


        # Filtro per trattamento superficiale
        surface_options = ["No", "Blued", "White zinc-plated", "Zinc-plated", "Painted"]
        df["Surface"] = df["Surface"].astype(str).str.strip().str.lower()
        selected_surface = st.selectbox("Surface Treatment", ["All"] + surface_options)

        if selected_surface != "All":
            df = df[df["Surface"] == selected_surface.lower()]

        st.markdown("""<hr style="margin: 10px 0; border: 0.5px solid #ccc;">""", unsafe_allow_html=True)
        st.write("")
        st.markdown("### ️ Size")


               
        # Creiamo una nuova colonna per mostrare DN + Pollici
        df["DN_Display"] = df["DN"].astype(str) + " - " + df["Pollici"].astype(str)

# Creiamo il filtro per DN + Pollici
        dn_options = ["All"] + sorted(df["DN_Display"].unique().tolist())
        dn_selected = st.selectbox("Size (DN)", dn_options, index=0)

# Applichiamo il filtro solo se non è "All"
        if dn_selected != "All":
            df = df[df["DN_Display"] == dn_selected]
            
        st.markdown("""<hr style="margin: 10px 0; border: 0.5px solid #ccc;">""", unsafe_allow_html=True)
        st.write("")
        st.markdown("### ️ Customizations")    
        
        # Casella di testo per personalizzazioni
        custom_text = st.text_input("Customizations (key words)", "")

        if custom_text:
            query = custom_text.lower().strip()
            query_embedding = model.encode(query, convert_to_tensor=True)
            parole_chiave_query = set(query.split())

    # Calcolo del punteggio solo sui risultati già filtrati
            df["Match TF-IDF"] = df["TF-IDF"].apply(
                lambda row: sum(row.get(p, 0) for p in parole_chiave_query if isinstance(row, dict))
            )
            df["Similarità"] = [util.pytorch_cos_sim(query_embedding, emb)[0].item() for emb in df["embedding"]]
            df["Score Totale"] = df["Similarità"] * 0.5 + df["Match TF-IDF"] * 0.4
            df = df.sort_values(by="Score Totale", ascending=False).head(10)


    # Limita i risultati per velocità di rendering
    numero_risultati = len(df)
    LIMITE_PREDEFINITO = 20

# Mostra il numero totale di risultati trovati
    st.info(f"🔎 Found {numero_risultati} products matching the filters")

# Checkbox per mostrare tutti i risultati
    mostra_tutti = st.checkbox("Show all results (⚠️ may slow down rendering)", value=False)

# Applica il limite solo se l'utente non vuole vedere tutto
    if not mostra_tutti:
        df = df.head(LIMITE_PREDEFINITO)
        st.caption(f"Displaying top {LIMITE_PREDEFINITO} results. Check the box above to show all.")

    
    with col_risultati:
        st.header("Results")

        if "prodotti_selezionati_filtri" not in st.session_state:
            st.session_state.prodotti_selezionati_filtri = []

        # Mostriamo tutti i prodotti filtrati
        for idx, row in df.iterrows():
            prodotto_id = f"{row['Material']}_{row['Series']}_{idx}"

            if f"check_filtri_{prodotto_id}" not in st.session_state:
                st.session_state[f"check_filtri_{prodotto_id}"] = False

            

            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                 
                 with col1:
                     selezionato = st.checkbox("Select", key=f"check_filtri_{prodotto_id}", value=st.session_state[f"check_filtri_{prodotto_id}"])

    



            with col2:
                    
                gross_price_usd = calcola_prezzo_usd(row["Gross Price"])  # Conversione in Dollari
                # Calcolo del prezzo scontato in dollari
                    

                st.write(f"🔹 **{row['Material']}** - {row['Body material']} - {row['Flangiatura']} - DN: {row['DN']}, Inches: {row['Pollici']}")
                st.markdown(f"📌 **{row['Short Description']}**")
                if temp_unit == "Fahrenheit":
    # Inversa conversione per visualizzare anche i °C
                    temp_min_c = round((row["Temperatura Min"] - 32) * 5/9)
                    temp_max_c = round((row["Temperatura Max"] - 32) * 5/9)
                    temp_min_f = round(row["Temperatura Min"])
                    temp_max_f = round(row["Temperatura Max"])
                    st.write(f"🌡️ **Temperature:** {temp_min_c}°C / {temp_min_f}°F | {temp_max_c}°C / {temp_max_f}°F | ⚙️ **Pressure:** {round(row['BAR'])} BAR / {round(row['PSI'])} PSI")
                else:
                    temp_min_c = round(row["Temperatura Min"])
                    temp_max_c = round(row["Temperatura Max"])
                    temp_min_f = round((row["Temperatura Min"] * 9/5) + 32)
                    temp_max_f = round((row["Temperatura Max"] * 9/5) + 32)
                    st.write(f"🌡️ **Temperature:** {temp_min_c}°C / {temp_min_f}°F | {temp_max_c}°C / {temp_max_f}°F | ⚙️ **Pressure:** {round(row['BAR'])} BAR / {round(row['PSI'])} PSI")
                st.markdown(f"{row['descrizione'].replace(chr(10), '<br>')}", unsafe_allow_html=True)
                if isinstance(gross_price_usd, float):
                    st.write(f"💰 **Gross Price (USD):** {gross_price_usd:.2f}$")
                else:
                    st.write(f"💰 **Gross Price (USD):** {gross_price_usd}")

                    

                # Cerca il PDF nella cartella pdf_disegni/
                pdf_filename = resource_path("pdf_disegni", f"{row['Series']}.pdf")
                if os.path.exists(pdf_filename):
                    with open(pdf_filename, "rb") as pdf_file:
                        st.download_button(label="📄 Download PDF drawing", data=pdf_file, file_name=f"{row['Series']}.pdf", mime="application/pdf", key=f"pdf_{row['Series']}_{idx}")
                else:
                    st.write("❌ PDF drawing not available.")

           
            prodotto = {
                "Material": row["Material"],
                "Short Description": row["Short Description"],
                "DN": row["DN"],
                "Pollici": row["Pollici"],
                "Gross Price (USD)": calcola_prezzo_usd(row["Gross Price"]),
                "descrizione": row["descrizione"],  # Aggiunta la descrizione completa
                "Series": row["Series"],
            }
                

                
 
            if selezionato:
                if prodotto not in st.session_state.prodotti_selezionati_filtri:
                        st.session_state.prodotti_selezionati_filtri.append(prodotto)
            else:
                if prodotto in st.session_state.prodotti_selezionati_filtri:
                    st.session_state.prodotti_selezionati_filtri.remove(prodotto)




        st.markdown("---")
          
        if st.button("🛒 Add selected products to cart", key="add_to_cart_filtri"):

            if "carrello" not in st.session_state:
                st.session_state.carrello = []

            if not st.session_state.prodotti_selezionati_filtri:  # Controlliamo se la lista è vuota
                st.warning("⚠️ Nessun prodotto selezionato! Seleziona almeno un prodotto prima di aggiungerlo al carrello.")
            else:
        # Aggiungiamo i prodotti selezionati al carrello
                st.session_state.carrello.extend(st.session_state.prodotti_selezionati_filtri)
        
        # Puliamo la lista dei prodotti selezionati per evitare duplicati
                st.session_state.prodotti_selezionati_filtri.clear()
        
                st.success("🛒 Add selected products to cart")



from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
import tempfile

def genera_pdf():
    # 1. Genera il contenuto dinamico
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf_path = tmpfile.name

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    y_position = height - 100  # Posizione iniziale

    total_net = 0
    net_prices_validi = 0

    for item in st.session_state.carrello:
        if y_position < 150:
            c.showPage()
            y_position = height - 100

        # Titolo
        c.setFont("Helvetica-Bold", 12)
        titolo = f"{item['Material']} - {item['Short Description']}"
        c.drawString(50, y_position, titolo)
        y_position -= 20

        # Descrizione
        c.setFont("Helvetica", 10)
        descrizione = simpleSplit(item.get('descrizione', 'Nessuna descrizione disponibile'), "Helvetica", 10, 400)
        for line in descrizione:
            if y_position < 50:
                c.showPage()
                y_position = height - 50
            c.drawString(50, y_position, line)
            y_position -= 15

        # DN + Pollici
        c.setFont("Helvetica", 10)
        c.drawString(50, y_position, f"DN: {item['DN']} | Pollici: {item['Pollici']}")
        y_position -= 20

        # Prezzi
        c.setFont("Helvetica-Bold", 11)
        gross = item.get("Gross Price (USD)", "On request")
        netto = item.get("Net Price (USD)", "On request")

        if isinstance(gross, (int, float)):
            c.drawString(50, y_position, f"💰 Gross price: {gross:.2f} USD")
        else:
            c.drawString(50, y_position, f"💰 Gross price: {gross}")
        y_position -= 20

        if isinstance(netto, (int, float)):
            c.drawString(50, y_position, f"💲 Net price: {netto:.2f} USD")
            total_net += netto
            net_prices_validi += 1
        else:
            c.drawString(50, y_position, f"💲 Net price: {netto}")
        y_position -= 30

    # Spazio finale
    if y_position < 100:
        c.showPage()
        y_position = height - 100

    # Totale offerta
    c.setFont("Helvetica-Bold", 12)
    y_position -= 10
    if net_prices_validi > 0:
        c.drawString(50, y_position, f"💲 Quotation net total: {total_net:.2f} USD")
    else:
        c.drawString(50, y_position, "💲 Quotation net total: On request")

    c.save()

    # 2. Applica il template PDF aziendale come sfondo
    template_path = resource_path("sfondo_template.pdf")
    pdf_finale = applica_template_sfondo(pdf_path, template_path)

    return pdf_finale





from PyPDF2 import PdfReader, PdfWriter

def applica_template_sfondo(pdf_contenuti_path, pdf_template_path):
    output = PdfWriter()
    contenuti = PdfReader(pdf_contenuti_path)
    sfondo = PdfReader(pdf_template_path)

    for pagina in contenuti.pages:
        pagina_template = PdfReader(pdf_template_path).pages[0]  # Template fisso
        pagina_template.merge_page(pagina)
        output.add_page(pagina_template)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        output_path = tmpfile.name
        with open(output_path, "wb") as f:
            output.write(f)
    
    return output_path
    
                

def visualizza_carrello():
    st.title("🛒 Cart")

    if "carrello" not in st.session_state or not st.session_state.carrello:
        st.write("Your cart is empty.")
        return

    st.header("Selected products summary")

    st.markdown("---")
    st.header("Discounts")
    sconto = st.number_input("Enter the discount (%)", min_value=0.0, max_value=100.0, value=0.0, key="sconto_carrello_input")
    extra_sconto = st.number_input("Enter the extra discount (%)", min_value=0.0, max_value=100.0, value=0.0, key="extra_sconto_carrello_input")

    total_gross = 0  # Totale lordo inizializzato
    total_net = 0    # Totale netto inizializzato

    carrello_copy = st.session_state.carrello.copy()

    total_gross = 0
    for idx, item in enumerate(st.session_state.carrello):
        st.markdown("---")  # Separatore tra i prodotti
      
        col1, col2, col3 = st.columns([0.6, 0.3, 0.1])
        with col1:
            st.write(f"🔹 **{item['Material']}** - {item['Short Description']} - DN: {item['DN']}, Pollici: {item['Pollici']}")
            st.markdown(f"{item.get('descrizione', 'Nessuna descrizione disponibile').replace(chr(10), '<br>')}", unsafe_allow_html=True)


   

        with col2:

            gross_price = item.get("Gross Price (USD)", "On request")

            if isinstance(gross_price, (int, float)):
                net_price1 = gross_price - (gross_price * (sconto / 100)) 
                net_price = net_price1 - (net_price1 * (extra_sconto / 100))
                item["Net Price (USD)"] = net_price
            else:
                net_price = "On request"
                item["Net Price (USD)"] = "On request"


            if isinstance(gross_price, (int, float)):
                st.write(f"💰 **Gross price:** {gross_price:.2f} USD")
            else:
                st.write(f"💰 **Gross price:** {gross_price}")

            if isinstance(net_price, (int, float)):
                st.write(f"💲 **Net price:** {net_price:.2f} USD")
            else:
                st.write(f"💲 **Net price:** {net_price}")


            if isinstance(gross_price, (int, float)):
                total_gross += gross_price

            if isinstance(net_price, (int, float)):
                total_net += net_price
      # Aggiungiamo al totale netto con sconto

        with col3:
            if st.button("🗑️", key=f"remove_{idx}"):
                st.session_state.carrello.remove(item)
                st.rerun()

        # Verifica se esiste un PDF associato e mostra il bottone per scaricarlo
        pdf_filename = resource_path("pdf_disegni", f"{item['Series']}.pdf")
        
        if os.path.exists(pdf_filename):
            with open(pdf_filename, "rb") as pdf_file:
                st.download_button(
                    label="📄 View PDF",
                    data=pdf_file,
                    file_name=f"{item['Series']}.pdf",
                    mime="application/pdf",
                    key=f"pdf_carrello_{idx}"
                )
        else:
            st.write("❌ PDF not available.")
    
    final_total = total_net

    # Campi per lo sconto nel carrello
    st.markdown("---")
    st.header("Discounts")
    

    st.write(f"💰 Total gross price: {total_gross:.2f} USD")
    st.write(f"### 🎯 **Total net price:** {final_total:.2f} USD")    

    if st.button("Empty cart"):
        st.session_state.carrello = []
        st.rerun()

    st.markdown("---")

    if st.button("📄 Export PDF"):
        pdf_file_path = genera_pdf()
        with open(pdf_file_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download PDF",
                data=pdf_file,
                file_name="Carrello.pdf",
                mime="application/pdf"
            )




if __name__ == "__main__":
    main()





