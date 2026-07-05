from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import VECTORSTORE_PATH

def get_chain():
    """Initialize RAG chain"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings,
        collection_name="city_eye_rules"
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    prompt = ChatPromptTemplate.from_template("""أنت مساعد متخصص في قواعد City Eye.

المعلومات:
{context}

السؤال: {question}

أجب بناءً على المعلومات أعلاه فقط.""")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

def ask(question: str):
    """Ask a question and get answer with sources"""
    chain, retriever = get_chain()
    docs = retriever.invoke(question)
    answer = chain.invoke(question)
    return answer, docs