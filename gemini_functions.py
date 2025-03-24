import os
from typing import List

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
    try:
        doc_ref = db.collection("chatHistory").document(user_id).collection("chatSessions").document(chat_session_id)
        doc = doc_ref.get()
        return doc.to_dict().get("history", []) if doc.exists else []
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

async def save_chat_history(user_id: str, chat_session_id: str, history: List[dict]):
    try:
        doc_ref = db.collection("chatHistory").document(user_id).collection("chatSessions").document(chat_session_id)
        doc_ref.set({"history": history})
    except Exception as e:
        print(f"Error saving chat history: {e}")

def history_to_types(history: List[dict]) -> List[types.Content]:
    return [types.Content(role=message["role"], parts=[types.Part.from_text(text=message["text"])]) for message in history]

async def send_message_to_gemini(message: str, history: List[dict], prompt: str) -> str:
    try:
        google_search_tool = Tool(google_search = GoogleSearch())
        content = history_to_types(history) + [types.Content(role="user", parts=[types.Part.from_text(text=message)])]
        response = client.models.generate_content(model="gemini-2.0-flash",
                                                config= GenerateContentConfig(
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

