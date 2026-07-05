import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data/rules.pdf"
VECTORSTORE_PATH = "vectorstore"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")