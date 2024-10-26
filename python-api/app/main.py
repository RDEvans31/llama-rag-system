import os
import tempfile
import time
from fastapi import  FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Any
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.llm.hugging_face_inference_llama3_2 import HuggingFaceInference
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

INDEX_NAME = os.getenv("PINECONE_INDEX", 'nomadic-llama')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
logger = logging.getLogger("uvicorn")

model = HuggingFaceInference()
embeddings = HuggingFaceEndpointEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
# embeddings = OllamaEmbeddings(base_url='http://127.0.0.1:11434',model='llama3.2')
pc = Pinecone(api_key=PINECONE_API_KEY)
parser = StrOutputParser()
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
async def upload_file(file: UploadFile = File(...), source: str = '') -> Any:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    content = await file.read()
    file_name = file.filename

    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:

        logger.info(f"Starting to load {file_name} into memory")
        temp_file.write(content)
        temp_file.flush() 

        loader = PyMuPDFLoader(temp_file.name)
    

        if source == '':
            doc_source = file_name
        else:
            doc_source = source

        pages = [Document(page_content=page.page_content.replace("\n", " "), metadata={**page.metadata, "source": doc_source}) for page in loader.load()]

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

        chunks = splitter.split_documents(pages)

        if INDEX_NAME not in existing_indexes:
            pc.create_index(

            name=INDEX_NAME,

            dimension=384, 

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
        temp_file.close()
    
    return {"message": message}


@app.get("/query")
async def test_llm(query: str) -> Any:
    vectorstore: PineconeVectorStore = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    template = """
            You are an AI system running on Llama3.2 language model. You are owned and run by NomadicLifter.

            You are an expert on answering questions on life, health, fitness, and tech, using context if available.

            If the question is asking for guidance, provide implementation details. Answer concisely.

            Context: {context}

            Question: {input}
        """
    prompt = PromptTemplate.from_template(template)

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = await rag_chain.ainvoke({'input': query})
    return {
        "answer": response["answer"],
        "context": response["context"]
    }



