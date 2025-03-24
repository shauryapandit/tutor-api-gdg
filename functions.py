import os
from typing import List, Optional

import requests
from fastapi import HTTPException
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, GoogleSearch, Tool

from auth import db
from prompts import FINANCIAL_SYSTEM_PROMPT, SCORE_PROMPT, SYSTEM_PROMPT

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env!")

MODEL = "gemini-2.0-flash"

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
            model=MODEL,
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


async def send_to_gemini(prompt_text: str) -> str:
    """Sends a request to Gemini AI and returns its response."""
    try:
        response = client.models.generate_content(model=MODEL, 
                                                  contents=[{"role": "user", "parts": [{"text": prompt_text}]}]
                                                  )
        return response.text.strip() if response.text else "No response received."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")

async def generate_unique_question(level: str, asked_questions: list) -> str:
    """Generates a unique question that has not been asked before."""
    prompt_text = f"""
    {SYSTEM_PROMPT}
    Difficulty Level: {level}

    **Previously Asked Questions:**
    {', '.join(asked_questions) if asked_questions else 'None'}

    Generate a new question that has not been asked before.
    """

    for _ in range(5):  # Try multiple times to avoid repetition
        new_question = await send_to_gemini(prompt_text)
        if new_question not in asked_questions:
            return new_question

    return "No unique question could be generated."

async def evaluate_answer(user_answer: str, question_topic: str, level: str) -> int:
    """Evaluates the user's answer and assigns a score."""
    prompt_text = f"""
    {SCORE_PROMPT}

    **Question:** {question_topic}
    **User's Answer:** {user_answer}
    **Difficulty Level:** {level}
    """

    response_text = await send_to_gemini(prompt_text)
    lines = response_text.split("\n")
    score_line = next((line for line in lines if "Score:" in line), "Score: 0")

    try:
        return int(score_line.split(":")[-1].strip())
    except ValueError:
        return 0  # Default score if parsing fails