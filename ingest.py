from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PDF_PATH

def load_and_split():
    """Load PDF and split into chunks"""
    print("📄 جاري تحميل الـ PDF...")
    loader = PyMuPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"✅ تم تحميل {len(pages)} صفحة")

    print("✂️ جاري تقسيم المحتوى...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(pages)
    print(f"✅ تم إنشاء {len(chunks)} chunk")
    
    return chunks