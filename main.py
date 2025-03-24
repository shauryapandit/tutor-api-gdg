import logging
import os
import uuid

import google.generativeai as genai
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from auth import (authenticate_with_firebase, db, get_firebase_user,
                  refresh_firebase_token)
from functions import (evaluate_answer, generate_chat_session_id,
                       generate_unique_question, load_chat_history,
                       save_chat_history, send_message_to_gemini,
                       send_to_gemini)
from models import (AnswerRequest, ChatRequest, ChatRequestImage, LoginRequest,
                    RefreshRequest, StartRequest)
from prompts import FINANCIAL_SYSTEM_PROMPT

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"],
    allow_credentials= True,
    allow_methods= ["*"],
    allow_headers= ["*"]
)


@app.get("/")
async def root():
    return {"Status": "Active"}

@app.post("/v1/start")
async def start_quiz(request: StartRequest, user_data: dict = Depends(get_firebase_user)):

    """
    Starts a new quiz session for the given user.

    Args:
        request: The JSON payload containing the desired difficulty level.

    Returns:
        A JSON response containing the session ID and the first question.

    Raises:
        HTTPException: If the difficulty level is invalid.
    """
    if request.level not in ["Beginner", "Intermediate", "Advanced"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty level")

    session_id = str(uuid.uuid4())
    user_id = user_data.get("uid")
    # Fetch a unique question
    generated_question = await generate_unique_question(request.level, [])

    session_data = {
        "userId": user_id,
        "sessionId": session_id,
        "level": request.level,
        "history": [],
        "askedQuestions": [generated_question],  # Track asked questions
        "currentQuestion": {"Topic": generated_question},
        "score": 0,
    }
    db.collection("quiz_sessions").document(user_id).collection("sessions").document(session_id).set(session_data)
    return {"sessionId": session_id, "message": generated_question}


@app.post("/v1/answer")
async def answer_question(request: AnswerRequest, user_data: dict = Depends(get_firebase_user)):
    """
    Processes the user's answer for the current quiz question, evaluates it, updates the quiz session, and provides the next question.

    Args:
        request (AnswerRequest): The request payload containing the user's answer and session ID.
        user_data (dict): The user data retrieved from Firebase authentication.

    Returns:
        dict: A JSON response containing the evaluation of the current answer, the next question, and the updated score.

    Raises:
        HTTPException: If the session does not exist.
    """

    uuid_user = user_data.get("uid")
    session_ref = db.collection("quiz_sessions").document(uuid_user).collection("sessions").document(request.sessionId)
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

@app.get("/v1/progress/{sessionId}")
def get_progress(sessionId: str, user_data: dict = Depends(get_firebase_user)):
    """
    Retrieves the progress of a given session.

    Args:
        sessionId (str): The unique ID of the session.
        user_data (dict): The user data retrieved from Firebase authentication.

    Returns:
        dict: A JSON response containing the history of questions and answers, and the current score.

    Raises:
        HTTPException: If the session does not exist.
    """
    userId = user_data.get("uid")
    session_ref = db.collection("quiz_sessions").document(userId).collection("sessions").document(sessionId)
    session_doc = session_ref.get()
    
    if not session_doc.exists:
        raise HTTPException(status_code=400, detail="No active session found")
    
    session_data = session_doc.to_dict()
    return {"history": session_data.get("history", []), "score": session_data.get("score", 0)}

@app.post("/v1/chatwithimage")
async def chat(request: ChatRequestImage, user_data: dict = Depends(get_firebase_user)):
    """
    Handles a chat request with an image and returns a response.

    Args:
        request (ChatRequestImage): The JSON payload containing the message, chat session ID, and image URL.
        user_data (dict): The user data retrieved from Firebase authentication.

    Returns:
        dict: A JSON response containing the response from Gemini AI and the chat session ID.

    Raises:
        HTTPException: If the chat session ID is invalid or if the user does not exist.
    """
    chat_session_id = request.chatSessionId or str(int(os.times()[4] * 1000))
    user_id = user_data.get("uid")
    history = await load_chat_history(user_id, chat_session_id)
    response = await send_message_to_gemini(request.message, request.imageUrl, history)
    new_history = history + [{"role": "user", "text": request.message, "image": request.imageUrl},
                             {"role": "model", "text": response}]
    await save_chat_history(user_id, chat_session_id, new_history)
    return {"reply": response, "chatSessionId": chat_session_id}

@app.post("/v1/login")
def login(request_data: LoginRequest):
    """
    Authenticates a user with Firebase using email and password.

    Args:
        request_data (LoginRequest): The request payload containing the user's email and password.

    Returns:
        dict: A JSON response containing the authentication tokens and expiration time.

    Raises:
        HTTPException: If authentication fails or an internal error occurs.
    """

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

@app.post("/v1/refresh")
def refresh_token(request_data: RefreshRequest):
    """
    Refreshes the Firebase authentication token.

    Args:
        request_data (RefreshRequest): The request payload containing the refresh token.

    Returns:
        dict: A JSON response containing the new authentication tokens and expiration time.

    Raises:
        HTTPException: If the refresh token is invalid or if an internal error occurs.
    """
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

@app.get("/v1/protected-route")
def protected_route(user_data: dict = Depends(get_firebase_user)):
    """
    A protected route that requires user authentication.

    Args:
        user_data (dict): The user data retrieved from Firebase authentication.

    Returns:
        dict: A JSON response containing a welcome message and the authenticated user's data.
    """

    return {"message": "Welcome to the protected route!", "user_data": user_data}


# Error Handling
@app.exception_handler(Exception)
def handle_exception(request, exc):
    print(f"Unhandled error: {exc}")
    return HTTPException(status_code=500, detail=str(exc))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
