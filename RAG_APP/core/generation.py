# RAG_APP/core/generation.py
from RAG_APP.processing.embeddings import chroma_db, query_db
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from prompts import SYSTEM_PROMPT
from config import Config

config = Config()
GOOGLE_API_KEY = config.GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

def get_rag_response(query: str) -> dict:

    retriever = chroma_db.as_retriever(search_kwargs={'k': 5})
    parser = StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    template = """
    Guided by the system prompt {system_prompt}
    Answer the question based only on the following context:
    {context}
    Question: {question}
    Answer: """

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": retriever | format_docs,
             "question": RunnablePassthrough(),
             "system_prompt": lambda _: SYSTEM_PROMPT,
    }      
        | prompt
        | llm
        | parser
    )

    answer = rag_chain.invoke(query)

    db_docs = query_db(query)

    sources = [doc.metadata.get("source") for doc in db_docs]
    
    return {
        "answer": answer,
        "sources": sources
    }