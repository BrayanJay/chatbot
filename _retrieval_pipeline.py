from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

#Load embeddings and vector store
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


query = "What is the total asset value ot Asia Asset Finance PLC as at 31 March 2025"

#retriever = db.as_retriever(search_kwargs={"k": 5})

retriever = db.as_retriever(
    search_type="Similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_treshold": 0.3 #Only return chunks with cosine similarity 0.3
    }
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

#Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Documents {i}:\n{doc.page_content}\n")