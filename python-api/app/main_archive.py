import json
from dotenv import load_dotenv
from fastapi import  FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from uuid import uuid4
from typing import Any
import io
import os
import time
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
from app.llm.hugging_face_inference_llama3_2 import HuggingFaceInference

# load_dotenv(".env")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", 'http://127.0.0.1:11434') 
MODEL = os.getenv("LLM", 'llama3.2')
INDEX_NAME = os.getenv("PINECONE_INDEX", 'nomadic-llama')


def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite+aiosqlite:///memory.db", async_mode=True)

logger = logging.getLogger("uvicorn")

print(MODEL, OLLAMA_URL, INDEX_NAME)

# model = ChatOllama(base_url=OLLAMA_URL,model=MODEL, temperature=0)
model = HuggingFaceInference()

embeddings = OllamaEmbeddings(base_url=OLLAMA_URL,model=MODEL)

os.environ["PINECONE_API_KEY"] = "faec4084-0024-486f-bcf8-3ca6f74bb688"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def hello_world():
    
    logger.info(f"Api running!")

    return {
        "status": 200,
        "body": "Api running."
    }

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)) -> Any:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    content = await file.read()
    

    # with io.open(file_location, 'wb') as out_file:
    #     content = await file.read()  # Read file content
    #     out_file.write(content)  # Save file locally

    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:

        temp_file.write(content)
        temp_file.flush() 

        loader = PyPDFLoader(temp_file.name)
        
        logger.info(f"Starting to load {file.filename} into memory")

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

            number_of_batches = len(chunks)//10
            if len(chunks) % 10 != 0:
                number_of_batches += 1

            # vectorstore = PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)    

            for batch_num, i in enumerate(range(0, len(chunks), 10), start=1):
                batch = chunks[i:i+10]
                logger.info(f"Upserting batch {batch_num}/{number_of_batches} into Pinecone index: {INDEX_NAME}")
                vectorstore.add_documents(batch) # need to change the source name in metadata.
                

            logger.info(f"Successfully upserted {len(chunks)} chunks into Pinecone index: {INDEX_NAME}")

        except Exception as e:
            logger.error(f"Error during upsert operation: {str(e)}")

            message = f"Error during upsert operation: {str(e)}"

        finally:
            logger.info(f"Upsert operation completed for Pinecone index: {INDEX_NAME}")
    
    return {"message": message}

@app.get("/query")
async def test_llm(query: str) -> Any:

    template = """
            You are an expert assistant owned and run by NomadicLifter.

            You are an expert on answering questions on life, health, fitness, and tech, using context if available.

            If the question is asking for guidance, provide implementation details.

            Question: {question}
        """
    prompt = PromptTemplate.from_template(template)
    chain = {"question": itemgetter("question")} | prompt | model
    answer = await chain.ainvoke({'question': query})
    return {
        "answer": answer
    }

@app.get("/search")
async def search(session_id: str , query: str) -> Any:
    vectorstore: PineconeVectorStore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    template = """
            You are an expert assistant answering questions based on the provided context.

            If the question is asking for guidance, provide implementation details.
            If the context doesn’t contain enough information, say "I don't know."

            Context: {context}

            Question: {input}
        """
    
    template = """
            You are an expert assistant at answering questions.

            If the question is asking for guidance, provide implementation details.

            Question: {input}
        """

    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
    )

    # qa_chain = create_stuff_documents_chain(model,prompt)

    # rag_chain: Runnable = create_retrieval_chain(retriever, qa_chain)

    # chain_with_history = RunnableWithMessageHistory(qa_chain, get_session_history, input_messages_key="question", history_messages_key="history")
    # response = await chain_with_history.ainvoke({'input': query}, config={"configurable": {"session_id": session_id}})
    response = model.invoke(query)


    # serialise the context
    # context = response["context"]

    # serialized_docs = [
    #     {
    #         "metadata": doc.metadata,
    #         "page_content": doc.page_content
    #     }
    #     for doc in context
    # ]

    return {
        "session_id": session_id,
        "answer": response

        # "answer": response["answer"]
        # "context": json.dumps(serialized_docs, indent=4),
    }



