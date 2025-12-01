from langchain_community. document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from langchain_ollama import OllamaEmbeddings


model = 'llama3.1'
doc_path = "./data/"

#-------PDF Ingestion

if doc_path:
    loader = DirectoryLoader(
        doc_path,
        glob="*.pdf",                 
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    # print(f"Loaded {len(documents)} pages from PDFs")
else:
    print("no docs found")

#------Extracting and splitting

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 300)
chunks = text_splitter.split_documents(documents)
print("done splitting...")

print(f"Number of chunks {len(chunks)}")

#-------Adding to Vector DB

ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model = "snowflake-arctic-embed"),
    persist_directory="./chromadb",
    collection_name="simple-rag"
)

print("done adding to vector db")

vector_db.persist()
print("Vector DB built & saved.")

#---------Retrieval
 
