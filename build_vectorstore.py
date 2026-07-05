from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from ingest import load_and_split
from config import VECTORSTORE_PATH

def build():
    """Build and persist vector store"""
    print("🔄 جاري بناء Vector Store...")
    chunks = load_and_split()
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH,
        collection_name="city_eye_rules"
    )
    
    print(f"✅ تم بناء Vector Store في: {VECTORSTORE_PATH}")

if __name__ == "__main__":
    build()