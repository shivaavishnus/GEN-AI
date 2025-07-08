import os
import pickle
from dotenv import load_dotenv
import redis
import openai

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_BASE")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_version = os.getenv("AZURE_OPENAI_VERSION")


# Custom embedding class
class AzureOpenAIEmbedding(Embeddings):
    def __init__(self):
        self.model = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

    def embed_documents(self, texts):
        response = openai.Embedding.create(input=texts, engine=self.model)
        return [d["embedding"] for d in response["data"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Instantiate embedding model
embedding_model = AzureOpenAIEmbedding()

# LangChain Azure Chat
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    openai_api_base=os.getenv("AZURE_OPENAI_BASE"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_type="azure",
    openai_api_version=os.getenv("AZURE_OPENAI_VERSION"),
)

# Redis connection
redis_client = redis.Redis.from_url(
    os.getenv("REDIS_URL"),
    decode_responses=False
)

# Load and split files
def load_and_split_documents(uploaded_files):
    documents = []
    for file in uploaded_files:
        path = f"/tmp/{file.name}"
        with open(path, "wb") as f:
            f.write(file.read())

        loader = PyPDFLoader(path) if file.name.endswith(".pdf") else TextLoader(path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(documents)

# Vector DB
def create_chroma_db(docs):
    persist_dir = "chroma_db"
    vectordb = Chroma.from_documents(docs, embedding_model, persist_directory=persist_dir)
    return vectordb.as_retriever()

# QA Chain
def create_qa_chain(db):
    return RetrievalQA.from_chain_type(llm=llm, retriever=db)

# Redis cache
def check_cache(question: str):
    key = f"rag_cache:{question}"
    return pickle.loads(redis_client.get(key)) if redis_client.exists(key) else None

def save_cache(question: str, answer: str):
    key = f"rag_cache:{question}"
    redis_client.set(key, pickle.dumps(answer))

def get_chat_history(session_id: str):
    key = f"chat_history:{session_id}"
    return pickle.loads(redis_client.get(key)) if redis_client.exists(key) else []

def save_chat_history(session_id: str, history):
    key = f"chat_history:{session_id}"
    redis_client.set(key, pickle.dumps(history))