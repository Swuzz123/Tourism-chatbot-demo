from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Create embedding function
def emb_texts(texts):
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="RETRIEVAL_DOCUMENT"
    )
    embeddings = embed_model.embed_documents(texts)
    return embeddings