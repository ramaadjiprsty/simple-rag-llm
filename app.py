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

        # memuat dokumen format pdf
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        # inisialisasi text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        # context = "\n".join(str(p.page_content) for p in pages)
        # texts = text_splitter.split_text(context)

        # memecah dokumen
        chunks = text_splitter.split_documents(pages)
        
        # buat vector store dari pecahan dokumen
        vector_store = Chroma.from_documents(documents=chunks, 
                                            embedding=embeddings,
                                            persist_directory=VECTOR_STORE_PATH)
        print("Vector store berhasil dibuat dan disimpan!")
    else:
        print("Memuat vector store yang sudah ada...")
        vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
        print("Vector store berhasil dimuat!")

    return vector_store.as_retriever(search_type="similarity_score_threshold",
                                    search_kwargs={"score_threshold": 0.7})


def get_conversational_chain():
    """
    Fungsi ini membuat QA chain menggunakan LCEL (LangChain Expression Language) dan mengimplementasi riwayat chat.
    """
    # prompt_template = """
    # Anda adalah asisten AI yang menjawab pertanyaan dari dokumen yang diberikan.
    # Jawab pertanyaan berdasarkan konteks yang diberikan. Pastikan untuk memberikan jawaban yang paling akurat. 
    # Jika jawaban tidak ditemukan dalam konteks, katakan, "Jawaban tidak tersedia dalam konteks". Jangan mengarang jawaban.\n\n
    # Konteks:\n {context}?\n
    # Pertanyaan: \n{question}\n

    # Jawaban:
    # """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""
            Anda adalah asisten AI ahli yang menjawab pertanyaan berdasarkan dokumen hukum.
            Gunakan hanya potongan-potongan konteks yang relevan di bawah ini untuk menjawab pertanyaan.
            Abaikan konteks yang berisi "Cukup jelas" atau yang tidak berhubungan dengan pertanyaan.
            Jika setelah mengabaikan konteks yang tidak relevan, jawaban masih tidak ditemukan, katakan, "Jawaban tidak tersedia dalam dokumen."
            Jangan mengarang jawaban.\n\n
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    model = init_chat_model(model="gemini-2.5-pro", temperature=0.3, model_provider="google_genai")
    # prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    output_parser = StrOutputParser() # Inisialisasi parser

    def format_docs(docs):
        # Menggabungkan isi dokumen menjadi satu string
        return "\n\n".join(doc.page_content for doc in docs)

    # Membangun rantai (chain) dengan LCEL
    chain = (
        # Input: {"input_documents": ..., "question": ...}
        # Output dari langkah pertama: {"context": ..., "question": ...}
        {"context": lambda x: format_docs(x["input_documents"]), "question": lambda x: x["question"], "chat_history": lambda x: x["chat_history"]}
        | prompt          # Masukkan ke prompt
        | model           # Kirim ke model LLM
        | output_parser   # Ambil output sebagai string
    )
    
    return chain


def main():
    pdf_path = "data/uu_lalin.pdf"
    print("--- Memulai Proses RAG di Terminal ---")
    
    if not os.path.exists(pdf_path):
        print(f"Error: File tidak ditemukan di path: {pdf_path}")
        return

    print(f"1. Memproses atau memuat vector store...")
    retriever = get_vector_store(pdf_path)

    print(f"\n2. Menyiapkan QA Chain")
    chain = get_conversational_chain()
    print("   Chain siap digunakan. Ketik Pertanyaan dan ketik exit untuk mengakhiri.")

    # inisialisasi list untuk riwayat chat
    chat_history = []

    while True:
        user_question = input("Pertanyaan: ")
        if user_question.lower() == "exit":
            print("Terima kasih! Aplikasi selesai.")
            break

        print("AI: Mencari jawaban...")
        docs = retriever.invoke(user_question)

        # menjalankan chain dengan riwayat chat
        response = chain.invoke({
            "input_documents": docs,
            "question": user_question,
            "chat_history": chat_history
        })

        print(f"AI: {response}")

        # menambahkan pertanyaan dan jawaban ke riwayat
        chat_history.append(HumanMessage(content=user_question))
        chat_history.append(AIMessage(content=response))

        # fungsi untuk menampilkan sumber (source citation)
        print("\n --- Sumber Jawaban ---")
        for i, doc in enumerate(docs):
            page_number = doc.metadata.get("page", "Tidak Diketahui")
            print(f" Sumber #{i+1} (Halaman: {page_number})")
            # mencetak 100 karakter pertama dari sumber
            print(f"'{doc.page_content[:100]}...'")
            print("------------------------------")

if __name__ == "__main__":
    main()