# app.py atau test_terminal.py (Versi Terbaru)

import os
from dotenv import load_dotenv
import google.generativeai as genai

# LangChain imports (dengan tambahan StrOutputParser)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser # Impor parser output

from langchain.chat_models import init_chat_model

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Muat environment variables dari file .env
load_dotenv()

# Konfigurasi API key Google
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API Key Google tidak ditemukan. Mohon buat file .env dan isikan GOOGLE_API_KEY.")
else:
    genai.configure(api_key=api_key)


def get_vector_store(pdf_path):
    """
    Fungsi ini membuat atau memuat vector store yang sudah ada.
    """
    VECTOR_STORE_PATH = "vector_store/"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if not os.path.exists(VECTOR_STORE_PATH):
        print("Membuat vector store baru dari PDF. Proses ini mungkin memakan waktu beberapa menit...")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        context = "\n".join(str(p.page_content) for p in pages)
        texts = text_splitter.split_text(context)
        
        vector_store = Chroma.from_texts(texts, embeddings, persist_directory=VECTOR_STORE_PATH)
        print("Vector store berhasil dibuat dan disimpan!")
    else:
        print("Memuat vector store yang sudah ada...")
        vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
        print("Vector store berhasil dimuat!")

    return vector_store.as_retriever(search_kwargs={"k": 5})


def get_conversational_chain():
    """
    Fungsi ini membuat QA chain menggunakan LCEL (LangChain Expression Language) dan mengimplementasi riwayat chat.
    """
    prompt_template = """
    Anda adalah asisten AI yang menjawab pertanyaan dari dokumen yang diberikan.
    Jawab pertanyaan berdasarkan konteks yang diberikan. Pastikan untuk memberikan jawaban yang paling akurat. 
    Jika jawaban tidak ditemukan dalam konteks, katakan, "Jawaban tidak tersedia dalam konteks". Jangan mengarang jawaban.\n\n
    Konteks:\n {context}?\n
    Pertanyaan: \n{question}\n

    Jawaban:
    """
    
    model = init_chat_model(model="gemini-2.5-pro", temperature=0.3, model_provider="google_genai")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    output_parser = StrOutputParser() # Inisialisasi parser

    def format_docs(docs):
        # Menggabungkan isi dokumen menjadi satu string
        return "\n\n".join(doc.page_content for doc in docs)

    # Membangun rantai (chain) dengan LCEL
    chain = (
        # Input: {"input_documents": ..., "question": ...}
        # Output dari langkah pertama: {"context": ..., "question": ...}
        {"context": lambda x: format_docs(x["input_documents"]), "question": lambda x: x["question"]}
        | prompt          # Masukkan ke prompt
        | model           # Kirim ke model LLM
        | output_parser   # Ambil output sebagai string
    )
    
    return chain


def main():
    pdf_path = "data/uu_lalin.pdf"
    pertanyaan_tes = "Sebutkan jenis-jenis surat izin mengemudi dan diatur dalam pasal berapa?"

    print("--- Memulai Proses RAG di Terminal ---")
    
    if not os.path.exists(pdf_path):
        print(f"Error: File tidak ditemukan di path: {pdf_path}")
        return

    print(f"1. Memproses atau memuat vector store...")
    retriever = get_vector_store(pdf_path)

    print(f"\n2. Mencari konteks relevan untuk pertanyaan...")
    docs = retriever.invoke(pertanyaan_tes)
    
    print("\n3. Menyiapkan QA chain dengan LCEL...")
    chain = get_conversational_chain()
    print("   Chain siap digunakan.")

    print("\n4. Mengirim permintaan ke Gemini...")
    # Menjalankan chain dengan input yang dibutuhkan
    response = chain.invoke({"input_documents": docs, "question": pertanyaan_tes})
    
    # Hasil 'response' sekarang adalah string langsung, bukan dictionary
    print("\n--- JAWABAN DARI GEMINI ---")
    print(response)
    print("--------------------------")


if __name__ == "__main__":
    main()