import os
import requests
from typing import List, Optional

from fastapi import HTTPException
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool

from auth import db
from prompts import FINANCIAL_SYSTEM_PROMPT

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))





def generate_chat_session_id():
    return f"{int(os.times()[4] * 1000)}_{os.urandom(8).hex()}"


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

