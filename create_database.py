from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]

# embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
embedding_model = MistralAIEmbeddings(model="mistral-embed")


vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding= embedding_model,
    persist_directory= "chroma_db"
)