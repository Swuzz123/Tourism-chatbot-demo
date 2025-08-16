from pymilvus import connections, Collection, utility
from dotenv import load_dotenv

load_dotenv()

# Connect to milvus
def connect_to_milvus():
    try: 
        connections.connect(host="localhost", port="19530")
        print("Connected to Milvus server")
    
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        raise
    
# Get the collection store vector embedding
def get_collection():
    COLLECTION_NAME = 'tourism_search'
    if not utility.has_collection(COLLECTION_NAME):
        raise ValueError(f"Collection {COLLECTION_NAME} does not exist")
    collection = Collection(COLLECTION_NAME)
    collection.load()
    return collection