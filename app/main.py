from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from fastapi import FastAPI, File, UploadFile, Query
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Memory saver for conversation context
checkpointer = MemorySaver()

# Define message state
class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]

# Define the assistant node
def assistant(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

# Define the document summarizer node
def summarize_document(state: MessagesState):
    doc_text = state["messages"][0][1]  # Assuming the document is passed as the first message
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(doc_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings = llm.generate_embeddings([doc.page_content for doc in documents])
    vector_store = FAISS.from_embeddings(embeddings, documents)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    summary = qa_chain.run("Summarize the document.")
    return {"messages": [("assistant", summary)]}

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("summarize", summarize_document)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "assistant")
builder.add_edge("assistant", END)

# Compile graph with memory
graph = builder.compile(checkpointer=checkpointer)

# Create FastAPI app
app = FastAPI()

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")
        return {"message": "Document uploaded successfully!", "content": text}
    except Exception as e:
        return {"error": str(e)}

@app.get("/chat/{query}")
def get_content(query: str, thread_id: str = Query(...)):
    """
    Handles user queries with dynamic thread ID for memory context.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke({"messages": [("user", query)]}, config)
        return result
    except Exception as e:
        return {"output": str(e)}

