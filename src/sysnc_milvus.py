import os
import pyodbc
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility 

# Load file .env
load_dotenv()

# ============== Connect SQL Server ==============

server = os.getenv('DB_SERVER')
database = os.getenv('DB_DATABASE')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

conn_str = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"Trusted_Connection=yes;"
    f"TrustServerCertificate=yes;"
)
print(f"Connection string: {conn_str}")

# Load data from SQL Server
try:
    conn = pyodbc.connect(conn_str)
    query = "SELECT * FROM destinations"
    data = pd.read_sql(query, conn)
    conn.close()
except Exception as e:
    print(f"Error connecting to SQL Server: {e}")
    exit(1)
    
print("Data columns:", data.columns.tolist())
print(data.head())

# Embed texts
data = data.fillna("")
data['combined_text'] = data.apply(
    lambda row: f"{row['Destination']} {row['State']} {row['Description']} {row['TouristAttractions']} {row['Activities']}",
    axis=1
)

def emb_texts(texts):
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="RETRIEVAL_DOCUMENT"
    )
    embeddings = embed_model.embed_documents(texts)
    return embeddings

# Set the collection name and dimension for the embeddings
COLLECTION_NAME = "tourism_search"
DIMENSION = 768

# ============== Connect to Milvus server (Docker) ==============
try:
    connections.connect(host="localhost", port="19530")  # Kết nối đến Milvus server
except Exception as e:
    print(f"Error connecting to Milvus: {e}")
    exit(1)

# Xóa collection nếu đã tồn tại
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
    
# ============== Create collection which includes the id, destinations, state, description, tourist attractions, activities, embedding. ==============

# 1. Define stuctured collection
fields = [
    FieldSchema(name='ID', dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name='Destination', dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name='State', dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name='Description', dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name='TouristAttractions', dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name='Activities', dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]

# 2. Add fields to schema
schema = CollectionSchema(fields=fields, description="Travel destinations")
collection = Collection(name=COLLECTION_NAME, schema=schema)

# ============== Create index and load collection ==============
try:
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
except Exception as e:
    print(f"Error creating index or loading collection: {e}")
    exit(1)

# ============== Insert data (no batch) ==============
try:
    embeddings = emb_texts(data['combined_text'].tolist())
    entities = [
        {
            "ID": int(row['ID']),
            "Destination": row['Destination'] or "",
            "State": row['State'] or "",
            "Description": row['Description'] or "",
            "TouristAttractions": row['TouristAttractions'] or "",
            "Activities": row['Activities'] or "",
            "embedding": emb
        }
        for row, emb in zip(data.to_dict('records'), embeddings)
    ]

    collection.insert(entities)
    collection.flush()  # Đồng bộ dữ liệu vào lưu trữ
    print(f"Inserted {len(entities)} entities into Milvus")
except Exception as e:
    print(f"Error inserting data into Milvus: {e}")
    exit(1)

# Kiểm tra số lượng entities
try:
    collection.load()  # Tải collection để truy vấn
    print(f"Total entities in Milvus: {collection.num_entities}")
except Exception as e:
    print(f"Error querying Milvus: {e}")
    exit(1)