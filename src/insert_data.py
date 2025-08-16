import pandas as pd
import os
import pyodbc
from dotenv import load_dotenv

load_dotenv()

PATH = "data/travel_data.csv"
data = pd.read_csv(PATH)

# SQL Server connection details
server = os.getenv('DB_SERVER')
database = os.getenv('DB_DATABASE')

# Connect SQL Server
conn_str = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"Trusted_Connection=yes;"
    f"TrustServerCertificate=yes;"
)
print(f"Connection string: {conn_str}")

try: 
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    insert_query = """
        INSERT INTO destinations (Destination, State, Description, TouristAttractions, Activities)
        VALUES (?, ?, ?, ?, ?)
    """
    
    for _, row in data.iterrows():
        cursor.execute(insert_query, (
            row['Destination'],
            row['State'],
            row['Description'],
            row['TouristAttractions'],
            row['Activities']
        ))
        
    conn.commit()
    print("Data Inserted Successfully!")
    
except Exception as e:
    print(f"An error occured: {e}")
    
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()
        
