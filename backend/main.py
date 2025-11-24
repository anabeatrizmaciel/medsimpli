import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import io
import base64
import re
import os
from pydantic import BaseModel
from collections import defaultdict
import csv

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

class Feedback(BaseModel):
    user_id: int
    index: int
    termo: str
    simplificado: str
    useful: bool

feedback_count = defaultdict(lambda: {"pos": 0, "neg": 0})

# --- armazenar feedbacks em memória
feedback_storage = []

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

    # MODO HÍBRIDO
    for i in range(len(df)):
        pos = feedback_count[i]["pos"]
        neg = feedback_count[i]["neg"]
        total = pos + neg

        if total == 0:
            continue  # ninguém avaliou ainda

        ratio = pos / total  # valor entre 0 e 1

        # muito bem avaliado
        if ratio >= 0.75 and total >= 3:
            sims[i] += 0.20    # aumenta similaridade

        # muito mal avaliado
        elif ratio <= 0.25 and total >= 3:
            sims[i] -= 0.20    # penaliza similaridade

        # levemente positivo
        elif ratio >= 0.60:
            sims[i] += 0.10

        # levemente negativo
        elif ratio <= 0.40:
            sims[i] -= 0.10

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

@app.get("/metricas")
def calcular_metricas():
    # Calcula as 03 metricas para cada usuário

    # Parâmetros fixos
    top_k = 3
    boost_strength = 0.4
    
    try:
        # gabarito (avaliacoes.csv)
        avaliacoes_path = "../avaliacoes.csv"
        if not os.path.isfile(avaliacoes_path):
            avaliacoes_path = "avaliacoes.csv"
        
        df_avaliacoes = pd.read_csv(avaliacoes_path)
        
        def normalizar_termo(termo):
            return str(termo).lower().strip()
        
        def mapear_termo_para_item_id(termo_recomendado, todos_item_ids):
            # Mapeia um termo do dataset para o item_id correspondente no avaliacoes.csv
            termo_norm = normalizar_termo(termo_recomendado)
            primeira_palavra = termo_norm.split()[0] if termo_norm else termo_norm
            
            for item_id in todos_item_ids:
                item_norm = normalizar_termo(item_id)
                if item_norm == termo_norm or item_norm == primeira_palavra:
                    return item_norm
                if primeira_palavra == item_norm:
                    return item_norm
            return None
        
        def conjuntos_usuario(avaliacoes, usuario_id):
            #"Retorna: (relevantes, todos_itens) para um usuário
            df_u = avaliacoes[avaliacoes["usuario_id"] == usuario_id].copy()
            # Marca itens relevantes (nota >= 4)
            df_u["relevante"] = df_u["nota"] >= 4
            # Normaliza os item_ids
            df_u["item_id_norm"] = df_u["item_id"].apply(normalizar_termo)
            relevantes = set(df_u[df_u["relevante"] == True]["item_id_norm"])
            todos_itens = set(df_u["item_id_norm"])
            return relevantes, todos_itens
        
        # Lista para armazenar métricas de cada usuário
        lista_metricas_usuarios = []
        
        for usuario_id in sorted(df_avaliacoes["usuario_id"].unique()):
            # Obter conjuntos do usuário (relevantes e todos os itens)
            relevantes, todos_itens = conjuntos_usuario(df_avaliacoes, usuario_id)
            
            if len(relevantes) == 0:
                # Se não tiver  itens relevantes, pula  usuário
                continue
            
            # Gerar recomendações para este usuário
            recomendados_set = set()
            avaliacoes_usuario = df_avaliacoes[df_avaliacoes["usuario_id"] == usuario_id]
            queries_usadas = set()
            
            for _, row in avaliacoes_usuario.iterrows():
                item_id = str(row["item_id"])
                query = item_id
                
                if query in queries_usadas:
                    continue
                queries_usadas.add(query)
                
                recs = recomendar(query, top_k, boost_strength)
                
                # Adiciona os termos recomendados ao conjunto
                for rec in recs:
                    termo_recomendado = rec["termo"]
                    item_id_mapeado = mapear_termo_para_item_id(termo_recomendado, todos_itens)
                    if item_id_mapeado:
                        recomendados_set.add(item_id_mapeado)
            
            # Identifica TP, FP, FN como conjuntos de itens
            itens_tp = list(recomendados_set & relevantes)  # relevantes que o sistema recomendou
            itens_fp = list(recomendados_set - relevantes)   # não relevantes que o sistema recomendou
            itens_fn = list(relevantes - recomendados_set)   # relevantes que o sistema NÃO recomendou
            
            tp = len(itens_tp)
            fp = len(itens_fp)
            fn = len(itens_fn)
            
            # Calcular métricas
            if len(recomendados_set) == 0:
                precision = 0.0
            else:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            # Adiciona métricas do usuário à lista (incluindo listas de itens)
            lista_metricas_usuarios.append({
                "usuario_id": int(usuario_id),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "itens_relevantes": sorted(list(relevantes)),  # itens que o usuário achou relevantes (nota >= 4)
                "itens_recomendados": sorted(list(recomendados_set)),  # itens que o sistema recomendou
                "itens_tp": sorted(itens_tp),  # acertos: relevantes E recomendados
                "itens_fp": sorted(itens_fp),  # falsos positivos: recomendados mas não relevantes
                "itens_fn": sorted(itens_fn)   # falsos negativos: relevantes mas não recomendados
            })
        
        # Calculo da média geral
        if len(lista_metricas_usuarios) > 0:
            df_metricas = pd.DataFrame(lista_metricas_usuarios)
            media = {
                "precision": round(df_metricas["precision"].mean(), 4),
                "recall": round(df_metricas["recall"].mean(), 4),
                "f1": round(df_metricas["f1"].mean(), 4),
                "num_usuarios": len(lista_metricas_usuarios)
            }
        else:
            media = {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "num_usuarios": 0
            }
        
        return {
            "por_usuario": lista_metricas_usuarios,
            "media": media
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao calcular métricas: {str(e)}")

@app.post("/feedback")
def save_feedback(fb: Feedback):

    # salva no CSV
    file_exists = os.path.isfile("feedback.csv")
    with open("feedback.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["user_id", "index", "termo", "simplificado", "useful"])
        writer.writerow([fb.user_id, fb.index, fb.termo, fb.simplificado, fb.useful])

    # registra para modo híbrido
    if fb.useful:
        feedback_count[fb.index]["pos"] += 1
    else:
        feedback_count[fb.index]["neg"] += 1

    return {"message": "Feedback recebido!"}