# rani_faiss_memory_temp_silent.py
import streamlit as st
import google.generativeai as genai
import numpy as np
import faiss
import os

# === KONFIGURASI DASAR ===
st.set_page_config(page_title="RANI", page_icon="💬", layout="centered")

# === API KEY GEMINI ===
GEMINI_API_KEY = "GEMINI_API_KEY"

if not GEMINI_API_KEY or GEMINI_API_KEY.strip() == "":
    st.error("❌ API Key Gemini belum diisi di variabel GEMINI_API_KEY.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

DOC_FILENAME = "sumber.txt"

# === ATUR TEMPERATUR MODEL ===
TEMPERATURE = 0.9  # 0.0 = faktual, 1.0 = kreatif


# === LOAD DOKUMEN SUMBER ===
if not os.path.exists(DOC_FILENAME):
    st.error(f"❌ File '{DOC_FILENAME}' tidak ditemukan.")
    st.stop()

with open(DOC_FILENAME, "r", encoding="utf-8") as f:
    sumber_teks = f.read()

paragraphs = [p.strip() for p in sumber_teks.split("\n\n") if p.strip()]


# === BUAT EMBEDDING ===
@st.cache_resource(show_spinner=False)
def buat_faiss_index(paragraphs):
    model = "models/gemini-embedding-exp-03-07"
    embeddings = []
    for para in paragraphs:
        try:
            emb = genai.embed_content(model=model, content=para)["embedding"]
            embeddings.append(emb)
        except Exception:
            embeddings.append(np.zeros(768))

    embeddings = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, paragraphs


index, embeddings, paragraphs = buat_faiss_index(paragraphs)


# === SEMANTIC SEARCH ===
def cari_konteks_semantik(query, index, paragraphs, top_k=3):
    query_emb = genai.embed_content(model="models/gemini-embedding-exp-03-07", content=query)["embedding"]
    query_emb = np.array([query_emb], dtype=np.float32)
    D, I = index.search(query_emb, top_k)
    hasil = "\n\n".join([paragraphs[i] for i in I[0]])
    return hasil


# === BUAT JAWABAN (DENGAN MEMORY + TEMPERATUR) ===
def jawab_gemini(pertanyaan, konteks, riwayat_chat):
    chat_history = "\n".join(
        [f"{'User' if r=='user' else 'RANI'}: {m}" for r, m in riwayat_chat[-5:]]
    )
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
Saya ingin Anda berperan sebagai dokumen yang sedang saya ajak bicara. Nama Anda "RANI - Asisten Layanan Infomrasi Pengadilan Agama Medan", dan Anda ramah, lucu, dan menarik. Gunakan konteks yang tersedia, jawab pertanyaan pengguna sebaik mungkin menggunakan sumber daya yang tersedia, dan selalu berikan pujian sebelum menjawab.
Jika tidak ada konteks yang relevan dengan pertanyaan yang diajukan, cukup katakan "Hmm, kayaknya kamu langsung datang aja deh ke Pengadilan Agama Medan" dan berhenti setelahnya. Jangan menjawab pertanyaan apa pun yang tidak berkaitan dengan informasi. Jangan pernah merusak karakter.
=== RIWAYAT CHAT ===
{chat_history}
=== DOKUMEN SUMBER ===
{konteks}
=== PERTANYAAN BARU ===
{pertanyaan}
Jawablah sopan, ringkas, dan mudah dimengerti. 
Jika informasi tidak ditemukan, jawab:
"Hmmm... kayaknya kamu langsung datang aja deh ke Pengadilan Agama Medan."
Tambahkan tawaran bantuan di akhir jawaban.
"""
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=4096
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Terjadi kesalahan saat menghubungi Gemini: {e}"


# === BOOTSTRAP + AVATAR + ANIMASI + DARK MODE + ENTER SEND ===
import datetime
from streamlit.components.v1 import html

# Deteksi waktu lokal (gelap setelah jam 18.00)
hour = datetime.datetime.now().hour
is_dark = hour >= 18 or hour <= 5

bg_color = "#121212" if is_dark else "#f8f9fa"
header_color = "#0d6efd" if not is_dark else "#1f6feb"
text_color = "#f1f1f1" if is_dark else "#212529"
bubble_user_bg = "#3aafa9" if is_dark else "#d1e7dd"
bubble_bot_bg = "#2e2e2e" if is_dark else "#e9ecef"
bubble_user_color = "#ffffff" if is_dark else "#0f5132"
bubble_bot_color = "#f1f1f1" if is_dark else "#212529"

st.markdown(f"""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
body {{
    background-color: {bg_color};
    font-family: "Poppins", sans-serif;
    color: {text_color};
}}
.chat-wrapper {{
    max-width: 700px;
    margin: 25px auto;
    border-radius: 15px;
    background-color: {'#1e1e1e' if is_dark else '#ffffff'};
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    height: 85vh;
}}
.chat-header {{
    background-color: {header_color};
    color: white;
    padding: 15px;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
}}
.chat-body {{
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}}
.chat-message {{
    display: flex;
    align-items: flex-end;
    margin-bottom: 12px;
    animation: fadeIn 0.4s ease-in;
}}
.chat-message.user {{
    flex-direction: row-reverse;
}}
.chat-avatar {{
    width: 38px;
    height: 38px;
    border-radius: 50%;
    overflow: hidden;
    margin: 0 8px;
    box-shadow: 0 0 5px rgba(0,0,0,0.1);
}}
.chat-avatar img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
}}
.chat-bubble {{
    max-width: 70%;
    padding: 10px 15px;
    border-radius: 18px;
    font-size: 15px;
    line-height: 1.4;
    animation: fadeIn 0.4s ease-in;
}}
.user .chat-bubble {{
    background-color: {bubble_user_bg};
    color: {bubble_user_color};
    border-radius: 18px 18px 0 18px;
}}
.bot .chat-bubble {{
    background-color: {bubble_bot_bg};
    color: {bubble_bot_color};
    border-radius: 18px 18px 18px 0;
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
</style>
<script>
function scrollToBottom() {{
    const chatBody = window.parent.document.querySelector('.chat-body');
    if (chatBody) {{
        chatBody.scrollTop = chatBody.scrollHeight;
    }}
}}
setTimeout(scrollToBottom, 500);
</script>
""", unsafe_allow_html=True)


# === CHAT STRUCTURE ===
# st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)
# st.markdown("<div class='chat-header'>💬 RANI - ASISTEN LAYANAN INFORMASI PENGADILAN AGAMA MEDAN</div>", unsafe_allow_html=True)
st.markdown("<div class='chat-body'>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

AVATAR_USER = "https://cdn-icons-png.flaticon.com/512/847/847969.png"
AVATAR_BOT = "https://cdn-icons-png.flaticon.com/512/4712/4712100.png"

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="chat-avatar"><img src="{AVATAR_USER}"></div>
            <div class="chat-bubble">{msg}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot">
            <div class="chat-avatar"><img src="{AVATAR_BOT}"></div>
            <div class="chat-bubble">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # end chat-body


# === INPUT TANPA PLACEHOLDER, ENTER UNTUK KIRIM ===
user_input = st.chat_input("", key="chat_input_field")

# JavaScript untuk hapus placeholder dan kirim pakai Enter
html_code = """
<script>
const t = window.parent.document.querySelector('textarea');
if (t) {
    t.placeholder = '';
    t.style.minHeight = '38px';
    t.style.fontSize = '15px';
    t.style.borderRadius = '10px';
    t.style.padding = '8px 10px';
    t.style.width = '95%';
    t.style.margin = '10px auto';
    t.style.display = 'block';
    t.style.backgroundColor = 'white';
    // Tekan Enter langsung kirim (tanpa Shift)
    t.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const btn = window.parent.document.querySelector('button[kind="secondaryFormSubmit"]');
            if (btn) btn.click();
        }
    });
}
</script>
"""
html(html_code, height=0)


# === PROSES CHAT ===
if user_input:
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("🤖 RANI sedang berpikir..."):
        konteks = cari_konteks_semantik(user_input, index, paragraphs)
        jawaban = jawab_gemini(user_input, konteks, st.session_state.chat_history)

    st.session_state.chat_history.append(("bot", jawaban))

    if "chat_input_field" in st.session_state:
        del st.session_state["chat_input_field"]

    st.rerun()
