import logging
import os
import uuid

import google.generativeai as genai
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException

from auth import (authenticate_with_firebase, db, get_firebase_user,
                  refresh_firebase_token)
from gemini_functions import (generate_chat_session_id, load_chat_history,
                              save_chat_history, send_message_to_gemini, history_to_types, download_image)
from models import (AnswerRequest, ChatRequest, LoginRequest, RefreshRequest,
                    StartRequest, ChatRequest_v2)
from prompts import FINANCIAL_SYSTEM_PROMPT, SCORE_PROMPT, SYSTEM_PROMPT

load_dotenv()

app = FastAPI()



# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env!")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


@app.get("/")
async def root():
    return {"Status": "Active"}

@app.post("/start")
async def start_quiz(request: StartRequest):

    if request.level not in ["Beginner", "Intermediate", "Advanced"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty level")

    session_id = str(uuid.uuid4())

    # Fetch a unique question
    generated_question = await generate_unique_question(request.level, [])

    session_data = {
        "userId": request.userId,
        "sessionId": session_id,
        "level": request.level,
        "history": [],
        "askedQuestions": [generated_question],  # Track asked questions
        "currentQuestion": {"Topic": generated_question},
        "score": 0,
    }
    db.collection("quiz_sessions").document(request.userId).collection("sessions").document(session_id).set(session_data)
    return {"sessionId": session_id, "message": generated_question}

async def send_to_gemini(prompt_text: str) -> str:
    """Sends a request to Gemini AI and returns its response."""
    try:
        response = model.generate_content(prompt_text)
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

@app.post("/answer")
async def answer_question(request: AnswerRequest):
    session_ref = db.collection("quiz_sessions").document(request.userId).collection("sessions").document(request.sessionId)
    session_doc = session_ref.get()
    
    if not session_doc.exists:
        raise HTTPException(status_code=400, detail="No active session found!")

    session = session_doc.to_dict()
    question = session["currentQuestion"]["Topic"]
    level = session["level"]

    evaluation = await send_to_gemini(f"Evaluate: {question}\nUser's answer: {request.answer}")


    score = await evaluate_answer(request.answer, question, level)
    session["score"] += score

    history_entry = {
        "question": question,
        "userAnswer": request.answer,
        "evaluation": evaluation,
        "score": score,
    }
    session["history"].append(history_entry)

    # Generate a new question, ensuring uniqueness
    asked_questions = session.get("askedQuestions", [])
    new_question = await generate_unique_question(level, asked_questions)
    asked_questions.append(new_question)

    session["currentQuestion"] = {"Topic": new_question}
    session["askedQuestions"] = asked_questions
    session_ref.set(session, merge=True)

    return {"evaluation": evaluation, "nextQuestion": new_question, "currentScore": session["score"]}

@app.get("/progress/{userId}/{sessionId}")
def get_progress(userId: str, sessionId: str):
    session_ref = db.collection("quiz_sessions").document(userId).collection("sessions").document(sessionId)
    session_doc = session_ref.get()
    
    if not session_doc.exists:
        raise HTTPException(status_code=400, detail="No active session found")
    
    session_data = session_doc.to_dict()
    return {"history": session_data.get("history", []), "score": session_data.get("score", 0)}

@app.post("/chat")
async def chat(request: ChatRequest, user_data: dict = Depends(get_firebase_user)):
    uuid = user_data.get("uid")
    chat_session_id = request.chatSessionId or generate_chat_session_id()
    history = await load_chat_history(uuid, chat_session_id)
    response = await send_message_to_gemini(request.message, history, FINANCIAL_SYSTEM_PROMPT)
    new_history = history + [{"role": "user", "text": request.message}, {"role": "model", "text": response}]
    await save_chat_history(uuid, chat_session_id, new_history)
    return {"reply": response, "chatSessionId": chat_session_id}

@app.post("/login")
def login(request_data: LoginRequest):
    try:
        result = authenticate_with_firebase(request_data.email, request_data.password)
        return {
            "id_token": result.get("idToken"),
            "refresh_token": result.get("refreshToken"),
            "expires_in": result.get("expiresIn"),
        }
    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/refresh")
def refresh_token(request_data: RefreshRequest):
    try:
        result = refresh_firebase_token(request_data.refresh_token)
        return {
            "id_token": result.get("id_token"),
            "refresh_token": result.get("refresh_token"),
            "expires_in": result.get("expires_in"),
        }
    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/protected-route")
def protected_route(user_data: dict = Depends(get_firebase_user)):
    return {"message": "Welcome to the protected route!", "user_data": user_data}

# Error Handling
@app.exception_handler(Exception)
def handle_exception(request, exc):
    print(f"Unhandled error: {exc}")
    return HTTPException(status_code=500, detail=str(exc))

@app.post("/chatwithimage")
async def chat(request: ChatRequest_v2):
    chat_session_id = request.chatSessionId or str(int(os.times()[4] * 1000))
    history = await load_chat_history(request.userId, chat_session_id)
    response = await send_message_to_gemini(request.message, request.imageUrl, history)
    new_history = history + [{"role": "user", "text": request.message, "image": request.imageUrl},
                             {"role": "model", "text": response}]
    await save_chat_history(request.userId, chat_session_id, new_history)
    return {"reply": response, "chatSessionId": chat_session_id}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
