import os
import requests
from typing import List, Optional

import firebase_admin
from fastapi import FastAPI, HTTPException
from firebase_admin import credentials, firestore
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool
from pydantic import BaseModel

# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

app = FastAPI()

FINANCIAL_SYSTEM_PROMPT = """
You are an AI assistant that summarizes financial information about companies and stocks to help users make better investment decisions.
Provide:
- P/E Ratio: The company's price-to-earnings ratio.
- Beta: Stock risk compared to the market.
- Dividend: If the stock provides high dividends.
Answer politely and do not respond to non-finance-related queries.
"""

class ChatRequest(BaseModel):
    userId: str
    message: str
    chatSessionId: Optional[str] = None
    imageUrl: Optional[str] = None

async def load_chat_history(user_id: str, chat_session_id: str) -> List[dict]:
    """Loads chat history from Firebase Firestore."""
    try:
        doc_ref = db.collection("chatHistory").document(user_id).collection("chatSessions").document(chat_session_id)
        doc = doc_ref.get()
        return doc.to_dict().get("history", []) if doc.exists else []
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

async def save_chat_history(user_id: str, chat_session_id: str, history: List[dict]):
    """Saves chat history in Firebase Firestore."""
    try:
        doc_ref = db.collection("chatHistory").document(user_id).collection("chatSessions").document(chat_session_id)
        doc_ref.set({"history": history})
    except Exception as e:
        print(f"Error saving chat history: {e}")

def history_to_types(history: List[dict]) -> List[types.Content]:
    """Converts chat history into Gemini-compatible format."""
    return [types.Content(role=msg["role"], parts=[types.Part.from_text(text=msg["text"])]) for msg in history]

def download_image(image_url: str) -> bytes:
    """Downloads an image from a URL and returns it as bytes."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error downloading image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image URL or failed to download.")

async def send_message_to_gemini(message: str, image_url: Optional[str], history: List[dict]) -> str:
    """Sends text and image (if provided) to Gemini API."""
    try:
        google_search_tool = Tool(google_search=GoogleSearch())
        content = history_to_types(history) + [types.Content(role="user", parts=[types.Part.from_text(text=message)])]

        if image_url:
            image_bytes = download_image(image_url)
            content[-1].parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
                system_instruction=FINANCIAL_SYSTEM_PROMPT
            ),
            contents=content
        )
        return response.text
    except Exception as e:
        print(f"Error communicating with Gemini API: {e}")
        raise HTTPException(status_code=500, detail="Failed to communicate with Gemini API.")

@app.post("/chat")
async def chat(request: ChatRequest):
    chat_session_id = request.chatSessionId or str(int(os.times()[4] * 1000))
    history = await load_chat_history(request.userId, chat_session_id)
    response = await send_message_to_gemini(request.message, request.imageUrl, history)
    new_history = history + [{"role": "user", "text": request.message, "image": request.imageUrl},
                             {"role": "model", "text": response}]
    await save_chat_history(request.userId, chat_session_id, new_history)
    return {"reply": response, "chatSessionId": chat_session_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)