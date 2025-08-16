from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from utils.milvus_utils import connect_to_milvus, get_collection
from models.chatbot_model import generate_answer, get_relevant_chunk, make_prompt, system_message, chat_history

app = FastAPI()

# Connect to Milvus 
connect_to_milvus()
collection = get_collection()

if not chat_history:
    chat_history = []

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        if not data or 'query' not in data:
            return JSONResponse({"error": "No query provided"}, status_code=400)
        
        query = data['query']
        
        # Take the context and create a prompt
        context = get_relevant_chunk(query, collection)
        prompt = make_prompt(query, context)
        
        # Generate answer
        answer = generate_answer(system_message, chat_history, prompt)
        return JSONResponse({"answer": answer})
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)