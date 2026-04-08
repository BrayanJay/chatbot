import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

#Load all the files
def load_documents(docs_path="assets"):
    print(f"Loading documents from {docs_path}...")

    #check if assets directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exists. Please create it and add your company files")
        
    #load all .pdf files from the directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    # loads as: list item = per page per pdf file (eg:- pdf1 pg1, pdf1 pg2, pdf1 pg3, ...) 
    documents = loader.load()

    #check if langchain docs exists in the documents list
    if len(documents) == 0:
        raise FileNotFoundError(f"There are no files exists in the {docs_path} directory. Please add your company documents and try again.")
        
    #To verify files loaded successfully
    for i, doc in enumerate(documents[:5]): #use [:5] to show only first 2 pages of each pdf file
        print(f"\nDocument {i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")

    return documents

#To chunk the loaded file data
def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    """Split documents into smaller chunks with overlap"""
    print(f"Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks [:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"*50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

#To embedding and storing into vector db
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vectore store"""
    print("Creating embeddings and storing in ChromaDB")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    #Create ChromeDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    print("Main function")

    #1. Loading the files
    documents = load_documents(docs_path="assets")

    #2. Chunking the files
    chunks = split_documents(documents)

    #3. Embedding and storing in Vector DB
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()