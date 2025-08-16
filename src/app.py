import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from utils.milvus_utils import connect_to_milvus, get_collection
from models.chatbot_model import chat_history, system_message
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

load_dotenv()
    
# Create embedding function
async def emb_texts(texts):
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="RETRIEVAL_DOCUMENT"
    )
    embeddings = await asyncio.to_thread(embed_model.embed_documents, texts)
    return [float(x) for x in embeddings[0]] if embeddings and len(embeddings) > 0 else []

# Query similar embedding with vector embeddings in Milvus
async def get_relevant_chunk(query, collection):
    # Generate embedding for the query
    query_embedding = await emb_texts([query])
    
    # Search by top-k
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = await asyncio.to_thread(
        collection.search,
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=1,
        output_fields=["Destination", "State", "Description", "TouristAttractions", "Activities"]
    )
    
    if results and len(results[0]) > 0:
        result = results[0][0]  # Take the first entity
        context = (
            f"Destination: {result.entity.get('Destination')}\n"
            f"State: {result.entity.get('State')}\n"
            f"Description: {result.entity.get('Description')}\n"
            f"Tourist Attractions: {result.entity.get('TouristAttractions')}\n"
            f"Activities: {result.entity.get('Activities')}"
        )
        return context
    else:
        return "No relevant search found in the dataset."
    
def make_prompt(query, context):
    return f"Query: {query}\n\nContext:\n{context}\n\Answer: Please provide a warm, conversational response focusing on recreational activities if asked, using the context provided."

# Generate answer function
async def generate_answer(system_message, chat_history, prompt):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    chat_history.append(f"User: {prompt}")
    # Historical limit of 5 entries (User + Assistant)
    if len(chat_history) > 10:                  
        chat_history = chat_history[-10:]
        
    full_prompt = f"{system_message}\n\n" + "\n".join(chat_history) + "\nAssistant:"
    response = await model.generate_content_async(full_prompt)
    text_response = response.text.strip()
    chat_history.append(f"Assistant: {text_response}")
    return text_response
    
# =========== Streamlit UI ===========
st.title("Travel chatbot in India")

# Initialize session state and connect to Milvus
if "messages" not in st.session_state:
    st.session_state.messages = []
    connect_to_milvus()
    st.session_state.collection = get_collection()
    
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# Query input
query = st.chat_input("Ask anything...")

if query:
    # Display question of users
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Handle query and answer the user's question
    try:
        context = asyncio.run(get_relevant_chunk(query, st.session_state.collection))
        prompt = make_prompt(query, context)
        answer = asyncio.run(generate_answer(system_message, chat_history, prompt))
        
    except Exception as e:
        answer = f"Error: {str(e)}"
        
    # Dispay the answer
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})