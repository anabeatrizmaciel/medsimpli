# Geberal purpose library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import io
import base64
import re
import os

# Scikit-learn libraries
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

# Backend api library
from fastapi import FastAPI, HTTPException


app = FastAPI()

# ---- Funções auxiliares ----
def load_dataset():
    df = pd.read_csv("../dados_saude_com_bulas.csv")

    expected_cols = {"termo", "tecnico", "simplificado"}
    if not expected_cols.issubset(df.columns):
        raise HTTPException(status_code=404, detail="CSV inválido. As colunas obrigatórias são: termo, tecnico, simplificado.")
    
    return df

def build_vectorizer():
    portuguese_stopwords = list(text.ENGLISH_STOP_WORDS.union([
        "de","da","do","das","dos","em","para","por","com",
        "que","como","ou","uma","um","uns","umas","ao","aos",
        "na","nas","no","nos","e","o","a","os","as","se"
    ]))
    return TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words=portuguese_stopwords,
        ngram_range=(1,2),
        min_df=1
    )

def explain_contribution(query_vec, doc_vec, feature_names, top_k=5):
    q = query_vec.toarray()[0]
    d = doc_vec.toarray()[0]
    contrib = q * d
    idx = np.argsort(contrib)[::-1]
    return [(feature_names[i], float(contrib[i])) for i in idx[:top_k] if contrib[i] > 0]

def mapa_semantico_interativo(df):
    termos = df["termo"].astype(str).tolist()
    explicacoes = df["simplificado"].astype(str).tolist()

    # Vetorização
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),
        min_df=1
    )
    X = vectorizer.fit_transform(termos).toarray()

    # Redução de dimensionalidade - SOMENTE TSNE
    reducer = TSNE(
        n_components=2,
        perplexity=10,
        learning_rate=150,
        max_iter=1500,
        random_state=42
    )
    coords = reducer.fit_transform(X)

    # Normalização para não ficar enorme
    coords = StandardScaler().fit_transform(coords)
    coords = np.round(coords, 2)   # hover mais bonito

    # Clustering automático (até 5 grupos)
    k = min(5, len(df))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(coords)

    df_plot = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Termo": termos,
        "Explicação": explicacoes,
        "Cluster": clusters
    })

    return df_plot

# ---- Variáveis Globais ----
df = load_dataset()
vectorizer_tecnico = build_vectorizer()
vectorizer_simplif = build_vectorizer()
tfidf_tecnico = vectorizer_tecnico.fit_transform(df["tecnico"].astype(str))
tfidf_simplif = vectorizer_simplif.fit_transform(df["simplificado"].astype(str))

# ---- Rotas -----
@app.get("/")
def home():
    return {"message": "API de recomendação funcionando"}

@app.get("/load_data")
def load_data():
    df = load_dataset()
    
    return df.to_dict(orient="records")

@app.get("/get_debug_accuracy")
def debug_accuracy():
    acertos = 0
    for termo, esperado in zip(df['termo'], df['simplificado']):
        query_vec = vectorizer_tecnico.transform([termo])
        sims = cosine_similarity(query_vec, tfidf_tecnico)[0]
        pred = df.iloc[np.argmax(sims)]['simplificado']
        acertos += (pred == esperado)
    acuracia = acertos / len(df)

    return {"acc": acuracia}

@app.get("/recomendar_simplificacoes")
def recomendar(query, top_k: int, boost_strength: float):
    query_vec_tecnico = vectorizer_tecnico.transform([query])
    query_vec_simplif = vectorizer_simplif.transform([query])

    sims_tecnico = cosine_similarity(query_vec_tecnico, tfidf_tecnico)[0]
    sims_simplif = cosine_similarity(query_vec_simplif, tfidf_simplif)[0]
    sims = 0.5 * sims_tecnico + 0.5 * sims_simplif

    # --- Boost semântico ---
    for i, termo in enumerate(df["termo"]):
        termo_lower = termo.lower()
        query_lower = query.lower()
        if termo_lower == query_lower:
            sims[i] += boost_strength
        elif query_lower in termo_lower or termo_lower in query_lower:
            sims[i] += boost_strength / 2
    
    idxs = np.argsort(sims)[::-1][:top_k]

    resultado = []

    for rank, i in enumerate(idxs, start=1):
        termo = df.iloc[i]["termo"]
        simplif = df.iloc[i]["simplificado"]
        score = sims[i] * 100

        resultado.append({
            "rank": rank,
            "index": int(i),
            "termo": termo,
            "simplificado": simplif,
            "score": round(score, 2)
        })

    return resultado

@app.get("/termos_semelhantes")
def termos_semelhantes(index: int):
    termo_tecnico = tfidf_tecnico[index]
    sims_to_doc = cosine_similarity(termo_tecnico, tfidf_tecnico)[0]
    similar_idx = np.argsort(sims_to_doc)[::-1][1:6]  # ignora ele mesmo
    similares = [df.iloc[j]["termo"] for j in similar_idx if sims_to_doc[j] > 0.15]

    return {"similares": similares}

@app.get("/explicabilidade")
def show_explicability(query, index: int):
    doc_vec_simplif = tfidf_simplif[index]
    feature_names = vectorizer_simplif.get_feature_names_out()
    top_terms = explain_contribution(vectorizer_simplif.transform([query]), doc_vec_simplif, feature_names)
    chips = " "
    if top_terms:
        chips = "  ".join(f"`{t}`" for t, _ in top_terms)
        chips_global = chips
    
    return {"chips": chips}

@app.get("/mapa_interativo")
def gera_mapa():
    df_plot = mapa_semantico_interativo(df)

    return df_plot.to_dict(orient="records")

@app.get("/analisar_termos_dificeis")
def analisar_termos(text_input):
    texto_limpo = re.sub(r"[.,;:!?()]", " ", text_input.lower())
    matched = [(t, s) for t, s in zip(df["termo"], df["simplificado"]) if t.lower() in texto_limpo]

    return {"matched": matched}

@app.get("/substituir_termos")
def substituir_termos(texto):
    for termo, explic in zip(df["termo"], df["simplificado"]):
        texto = texto.replace(termo, f"{explic} ({termo})")
        texto = texto.replace(termo.lower(), f"{explic} ({termo})")

    return {"texto_out": texto}
