from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_chain import ask

app = FastAPI(
    title="City Eye RAG API",
    description="API للإجابة عن أسئلة قواعد City Eye",
    version="1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class SourceDoc(BaseModel):
    page: str
    content: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDoc] = []

# Endpoints

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "app": "City Eye Rules RAG API",
        "endpoints": {
            "POST /query": "اسأل سؤال",
            "GET /health": "فحص الحالة",
            "GET /docs": "Swagger Documentation"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Main query endpoint"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="السؤال فارغ")
        
        answer, docs = ask(request.question)
        
        sources = [
            {
                "page": str(doc.metadata.get("page", "?")),
                "content": doc.page_content[:200]
            }
            for doc in docs
        ]
        
        return {
            "question": request.question,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)