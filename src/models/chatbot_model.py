import google.generativeai as genai
from utils.embedding_utils import emb_texts
import os

chat_history = []

system_message = (
    "You are a fiendly and knowledageable travel assistant. "
    "Your answer questions only about destinations, their location (state), descriptions, tourist attractions and and recreational activities during that trip " 
    "based on the provided travel dataset."
    "If a query does not have an exact match in the data, provide the closest relevant information available. "
    "Use a warm, conservational tone, as if you are chatting with someone planning a trip. "
    "If the user asks about topics unrelated to travel or outside the dataset, report with: "
    "'I can only provide answers related to the travel destination I know about, specifically tourist spots in India.'"
)

# Generate answer function
def generate_answer(system_message, chat_history, prompt):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    chat_history.append(f"User: {prompt}")
    # Historical limit of 5 entries (User + Assistant)
    if len(chat_history) > 10:                  
        chat_history = chat_history[-10:]
        
    full_prompt = f"{system_message}\n\n" + "\n".join(chat_history) + "\nAssistant:"
    response = model.generate_content(full_prompt).text
    chat_history.append(f"Assistant: {response}")
    return response

# Query similar embedding with vector embeddings in Milvus
def get_relevant_chunk(query, collection):
    
    # Generate embedding for the query
    query_embedding = emb_texts([query])[0]
    
    # Search by top-k
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
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
    