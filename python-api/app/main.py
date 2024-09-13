import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from fastapi import  FastAPI, File, UploadFile, HTTPException
import logging
from uuid import uuid4
from typing import Any
import io
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from operator import itemgetter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables import Runnable


from langchain_community.retrievers import PubMedRetriever

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

MODEL = "llama3.1"
# MODEL = "gemma:2b"
INDEX_NAME = "nomadic-llama"
UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)
model = ChatOllama(model=MODEL, temperature=0)
embeddings = OllamaEmbeddings(model=MODEL)
logger = logging.getLogger("uvicorn")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

session_histories = {}

app = FastAPI()

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]


@app.get("/")
def hello_world():
    
    logger.info(f"Hello World!")

    return {
        "status": 200,
        "body": "Hello World"
    }

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)) -> Any:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    file_location = f"{UPLOAD_DIR}{file.filename}"
    with io.open(file_location, 'wb') as out_file:
        content = await file.read()  # Read file content
        out_file.write(content)  # Save file locally


    loader = PyPDFLoader(file_location)
    
    logger.info(f"Starting to load {file_location} into memory")

    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

    chunks = splitter.split_documents(pages)

    if INDEX_NAME not in existing_indexes:
        pc.create_index(

        name=INDEX_NAME,

        dimension=4096, # Replace with your model dimensions

        metric="cosine", # Replace with your model metric

        spec=ServerlessSpec(

            cloud="aws",

            region="us-east-1"

        ) 
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

    vectorstore: PineconeVectorStore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)

    logger.info(f"Index {INDEX_NAME} ready.")

    logger.info(f"Embeddings generated.")

    message = 'Successfully uploaded.'

    try:
        logger.info(f"Starting to upsert {len(chunks)} chunks into Pinecone index: {INDEX_NAME}")

        # await async_process_chunks_in_batches(chunks, embeddings)
        number_of_batches = len(chunks)//10
        if len(chunks) % 10 != 0:
            number_of_batches += 1

        # vectorstore = PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)    

        for batch_num, i in enumerate(range(0, len(chunks), 10), start=1):
            batch = chunks[i:i+10]
            logger.info(f"Upserting batch {batch_num}/{number_of_batches} into Pinecone index: {INDEX_NAME}")
            vectorstore.add_documents(batch)
            

        logger.info(f"Successfully upserted {len(chunks)} chunks into Pinecone index: {INDEX_NAME}")

    except Exception as e:
        logger.error(f"Error during upsert operation: {str(e)}")

        message = f"Error during upsert operation: {str(e)}"

    finally:
        logger.info(f"Upsert operation completed for Pinecone index: {INDEX_NAME}")
    
    return {"message": message}

@app.get("/search")
async def search(session_id: str , query: str) -> Any:
    vectorstore: PineconeVectorStore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    template = """
            You are an expert assistant answering questions based on the provided context.

            Answer the question directly and concisely.
            If the question is asking for guidance, provide implementation details.
            If the context doesn’t contain enough information, say "I don't know."

            Context: {context}

            Question: {input}
        """

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
    )

    qa_chain = create_stuff_documents_chain(model,prompt)

    rag_chain: Runnable = create_retrieval_chain(retriever, qa_chain)

    chain_with_history = RunnableWithMessageHistory(rag_chain, get_session_history, input_messages_key="question", history_messages_key="history")
    response = await chain_with_history.invoke({'input': query}, config={"configurable": {"session_id": session_id}})

    # serialise the context
    context = response["context"]

    serialized_docs = [
        {
            "metadata": doc.metadata,
            "page_content": doc.page_content
        }
        for doc in context
    ]

    return {
        "session_id": session_id,
        "answer": response["answer"],
        "context": json.dumps(serialized_docs, indent=4),
    }



