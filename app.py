import streamlit as st
from google import genai
from google.genai import types
import faiss
import numpy as np
from docx import Document
from pypdf import PdfReader

# Cấu hình trang
st.set_page_config(page_title="SmartDoc AI RAG", layout="wide")
st.title("🤖 Trợ lý Phân tích Tài liệu SmartDoc AI")

# Khởi tạo bộ nhớ tạm
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Chỉ dẫn thông minh giúp AI trả lời tự nhiên và linh hoạt
SYSTEM_PROMPT = """
Bạn là trợ lý AI thông minh, thân thiện, hỗ trợ giải đáp mọi câu hỏi và phân tích tài liệu.
Quy tắc trả lời:
1. NẾU CÓ TÀI LIỆU ĐÍNH KÈM: Ưu tiên tìm thông tin trong phần [Ngữ cảnh tài liệu]. Nếu tìm thấy, hãy trả lời chi tiết, rõ ràng và ghi trích dẫn [Nguồn: tên file - Trang X].
2. KHI KHÔNG CÓ TRONG TÀI LIỆU HOẶC CHƯA TẢI FILE: Đừng từ chối cứng nhắc. Hãy nói ngắn gọn "Tài liệu không đề cập trực tiếp, nhưng theo kiến thức chung..." rồi giải đáp tận tình cho người dùng.
3. KHI CÂU HỎI MƠ HỒ: Nếu người dùng hỏi quá ngắn (ví dụ: "đoạn 1", "mục này"), hãy dựa vào lịch sử trò chuyện để đoán ý hoặc lịch sự hỏi lại xem họ muốn tóm tắt hay giải thích phần nào.
"""

# Thanh cài đặt bên trái
st.sidebar.title("Cấu hình")
api_key = st.sidebar.text_input("Nhập Gemini API Key:", type="password")
uploaded_files = st.sidebar.file_uploader("Tải tài liệu lên (PDF, DOCX):", accept_multiple_files=True)

def extract_text(files):
    text_data = []
    for f in files:
        if f.name.endswith(".pdf"):
            pdf = PdfReader(f)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    text_data.append({"text": text, "source": f"{f.name} - Trang {i+1}"})
        elif f.name.endswith(".docx"):
            doc = Document(f)
            full_text = [para.text for para in doc.paragraphs if para.text.strip()]
            text_data.append({"text": "\n".join(full_text), "source": f.name})
    return text_data

def chunk_documents(docs, size=500):
    chunks = []
    for doc in docs:
        words = doc["text"].split()
        for i in range(0, len(words), size):
            chunk_text = " ".join(words[i:i+size])
            chunks.append({"text": chunk_text, "source": doc["source"]})
    return chunks

def get_embed(client, texts):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )
    embeddings = [emb.values for emb in response.embeddings]
    return np.array(embeddings, dtype=np.float32)

if st.sidebar.button("Xử lý tài liệu"):
    if not api_key:
        st.sidebar.error("Vui lòng nhập API Key!")
    elif not uploaded_files:
        st.sidebar.error("Chưa chọn tài liệu!")
    else:
        with st.spinner("Đang đọc và phân tích tài liệu..."):
            client = genai.Client(api_key=api_key)
            raw_docs = extract_text(uploaded_files)
            chunks = chunk_documents(raw_docs)
            vecs = get_embed(client, [c["text"] for c in chunks])
            
            idx = faiss.IndexFlatL2(vecs.shape[1])
            idx.add(vecs)
            
            st.session_state.index = idx
            st.session_state.chunks = chunks
            st.sidebar.success(f"Đã xử lý xong {len(chunks)} đoạn dữ liệu!")

# Hiển thị lịch sử trò chuyện
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Khung nhập câu hỏi
if prompt := st.chat_input("Hỏi bất cứ điều gì hoặc hỏi về tài liệu..."):
    if not api_key:
        st.error("⚠️ Vui lòng nhập Gemini API Key ở thanh bên trái trước khi hỏi.")
    else:
        st.session_state.msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                client = genai.Client(api_key=api_key)
                
                # 1. Lấy lịch sử 6 tin nhắn gần nhất để AI nhớ ngữ cảnh
                history_text = ""
                for msg in st.session_state.msgs[-7:-1]:
                    role_name = "Người dùng" if msg["role"] == "user" else "Trợ lý"
                    history_text += f"{role_name}: {msg['content']}\n"

                # 2. Tìm trong tài liệu (nếu người dùng đã tải file lên)
                retrieved = []
                ctx = "Chưa có tài liệu nào được tải lên."
                if st.session_state.index is not None and len(st.session_state.chunks) > 0:
                    q_vec = get_embed(client, [prompt])
                    k = min(4, len(st.session_state.chunks))
                    _, I = st.session_state.index.search(q_vec, k=k)
                    retrieved = [st.session_state.chunks[i] for i in I[0]]
                    ctx = "\n\n".join([f"[Nguồn: {c['source']}]\n{c['text']}" for c in retrieved])

                # 3. Gửi toàn bộ ngữ cảnh cho Gemini
                full_prompt = f"""
[Lịch sử trò chuyện gần đây]
{history_text}

[Ngữ cảnh tài liệu]
{ctx}

[Câu hỏi hiện tại]
{prompt}
"""
                res = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.4,
                    )
                )
                st.markdown(res.text)
                
                # Hiển thị nguồn trích dẫn nếu có tài liệu
                if retrieved:
                    with st.expander("🔍 Xem trích dẫn gốc trong tài liệu"):
                        for c in retrieved:
                            st.info(f"**{c['source']}**:\n\n{c['text']}")
                            
                st.session_state.msgs.append({"role": "assistant", "content": res.text})