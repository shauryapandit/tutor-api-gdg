import os
from typing import List

import firebase_admin
from fastapi import Body, FastAPI, HTTPException
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

# TODO: Move this to a config file and improve the prompt.
FINANCIAL_SYSTEM_PROMPT = """
You are an ai assistant that summarises information about companies and stocks that help users make better financial investment planning decisions. Provide the following information:
P/E Ratio: Look for the company's price-to-earnings (P/E) ratio—the current share price relative to its per-share earnings.
Beta: A company's beta can tell you how much risk is involved with a stock compared with the rest of the market.
Dividend: If you want to park your money, invest in stocks with a high dividend.
Answer accordingly in a polite way.
Do not answer any other query about topics other than finance.
"""
class ChatRequest(BaseModel):
    userId: str
    message: str
    chatSessionId: str


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

# Endpoints
@app.post("/chat")
async def chat(request: ChatRequest):
    chat_session_id = request.chatSessionId or generate_chat_session_id()
    history = await load_chat_history(request.userId, chat_session_id)
    response = await send_message_to_gemini(request.message, history, FINANCIAL_SYSTEM_PROMPT)
    new_history = history + [{"role": "user", "text": request.message}, {"role": "model", "text": response}]
    await save_chat_history(request.userId, chat_session_id, new_history)
    return {"reply": response, "chatSessionId": chat_session_id}


# Error Handling
@app.exception_handler(Exception)
def handle_exception(request, exc):
    print(f"Unhandled error: {exc}")
    return HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)