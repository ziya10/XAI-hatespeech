import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
import shap
import plotly.express as px
import pandas as pd

# --- 1. AYARLAR ---
st.set_page_config(page_title="XAI-Hate Detection ", page_icon="🛡️", layout="wide")

MODEL_PATH = "models/" 
OPTIMAL_THRESHOLD = 0.44 

@st.cache_resource
def load_assets():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    pred_pipeline = pipeline(
        "text-classification", 
        model=model, 
        tokenizer=tokenizer, 
        return_all_scores=True
    )
    return tokenizer, model, pred_pipeline

tokenizer, model, pred_pipeline = load_assets()

# --- 2. PLOTLY & SHAP ANALİZ FONKSİYONU ---
def plot_shap_with_plotly(text):
    explainer = shap.Explainer(pred_pipeline)
    shap_values = explainer([text])
    
    # Modelde Index 0 = Toxic olduğu için 0'ı hedefliyoruz
    tokens = shap_values[0].data
    values = shap_values[0].values[:, 0] 

    df = pd.DataFrame({"Kelime": tokens, "Etki Skoru": values})
    df = df.sort_values(by="Etki Skoru", ascending=True)

    df["Yön"] = df["Etki Skoru"].apply(lambda x: "🚨 Toxic Artıran" if x > 0 else "✅ Normalleştiren")

    fig = px.bar(
        df, 
        x="Etki Skoru", 
        y="Kelime", 
        orientation="h",
        color="Yön",
        color_discrete_map={"🚨 Toxic Artıran": "#ef553b", "✅ Normalleştiren": "#636efa"},
        labels={"Etki Skoru": "Karara Etkisi (SHAP Value)", "Kelime": ""},
        title="Kelimelerin Karar Üzerindeki Etkisi (Toxic Sınıfı Analizi)"
    )
    
    fig.update_layout(
        showlegend=True, 
        height=max(400, len(tokens) * 35),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# --- 3. ARAYÜZ ---
st.title("🛡️ XAI-Hate: Explainable Hate Speech Detection")
st.markdown(f"**Model:** DistilBERT | **Hassasiyet Eşiği:** {OPTIMAL_THRESHOLD}")

user_input = st.text_area("Analiz edilecek metni girin:", height=100, placeholder="Analiz edilecek cümleyi buraya yazın...")

if st.button("Analiz Et ve Görselleştir"):
    if user_input.strip() == "":
        st.warning("Lütfen bir metin girin.")
    else:
        # --- TAHMİN ---
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
            # Index 0 = Toxic sınıfı
            toxic_prob = probs[0][0].item() 

        # Sonuç Paneli
        col_res, col_space = st.columns([1, 1])
        with col_res:
            st.subheader("📊 Tahmin Sonucu")
            if toxic_prob >= OPTIMAL_THRESHOLD:
                st.error(f"### 🚨 TOXIC (%{toxic_prob*100:.2f})")
                st.write("Bu metin topluluk kurallarını ihlal ediyor olabilir.")
            else:
                # Normal olasılığını doğru hesapla
                st.success(f"### ✅ NORMAL (%{(1 - toxic_prob)*100:.2f})")
                st.write("Metin güvenli görünüyor.")

        st.divider()

        # --- XAI GÖRSELLEŞTİRME ---
        st.subheader("🔍 Kelime Bazlı Karar Analizi (SHAP)")
        st.write("Modelin neden bu kararı verdiğini görün:")
        
        with st.spinner("Plotly grafiği hazırlanıyor..."):
            try:
                fig = plot_shap_with_plotly(user_input)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Görselleştirme hatası: {e}")