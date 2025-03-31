import logging
import os
import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import firestore

from auth import (authenticate_with_firebase, db, get_firebase_user,
                  refresh_firebase_token)
from functions import (evaluate_answer, generate_unique_question,
                       load_chat_history, save_chat_history,
                       send_message_to_gemini, send_to_gemini)
from models import (AnswerRequest, ChatRequestImage, LoginRequest, Portfolio,
                    RefreshRequest, StartRequest)
from prompts import FINANCIAL_SYSTEM_PROMPT

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"Status": "Active"}

@app.post("/v2/profile")
async def portfolio(request: Portfolio, user_data: dict = Depends(get_firebase_user)):
    user_id = user_data.get("uid")
    user_ref = db.collection("profile").document(user_id)

    # user_ref.set({"Name": request.name,"Age": request.age,"Income": request.income,"Portfolio":firestore.ArrayUnion(request.portfolio)},merge=True)
    user_ref.set({"Name": request.name,"Age": request.age,"Income": request.income,"Portfolio":request.portfolio},merge=True)
    # Fetch the updated profile to return
    updated_profile = user_ref.get().to_dict()

    return {
        "message": "Profile successfully created or updated.",
        "profile": updated_profile
    }

@app.post("/v2/start")
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
    
    generated_question, topic = await generate_unique_question(request.level, [])

    session_data = {
        "userId": user_id,
        "sessionId": session_id,
        "level": request.level,
        "history": [],
        "askedQuestions": [generated_question],
        "askedTopics": [topic],
        "currentQuestion": {"Topic": generated_question},
        "score": 0,
    }
    # change it to a user profile
    db.collection("quiz_sessions").document(user_id).collection("sessions").document(session_id).set(session_data)
    # # Store asked question separately under userId -> askedQuestions
    # question_ref = db.collection("Topics").document("AskedTopics")
    # question_ref.set({"Topics": firestore.ArrayUnion([{"Topic": topic}])}, merge=True)

    return {"sessionId": session_id, "message": generated_question}

@app.post("/v2/answer")
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
    topic = session["askedTopics"]
    level = session["level"]

    evaluation = await send_to_gemini(f"Evaluate: {question}\nUser's answer: {request.answer}")
    score = await evaluate_answer(request.answer, question, level)
    
    # Update the Topics if it is known by user.
    if score > 0 and level == "Beginner":
        session["score"] += score
        # Store the topic if the answer is correct
        question_ref = db.collection("topics").document("AskedTopics")
        question_ref.set({"Topics": firestore.ArrayUnion([{"Topic": topic}])}, merge=True)
    if score > 0 and level == "Intermediate":
        session["score"] += score
        question_ref = db.collection("topics").document("AskedTopics")
        question_ref.set({"Topics": firestore.ArrayUnion([{"Topic": topic}])}, merge=True)
    if score > 1 and level == "Advanced":
        session["score"] += score
        question_ref = db.collection("topics").document("AskedTopics")
        question_ref.set({"Topics": firestore.ArrayUnion([{"Topic": topic}])}, merge=True)
    if score == 0 and level == "Advanced":
        session["score"] += score

    history_entry = {
        "question": question,
        "userAnswer": request.answer,
        "evaluation": evaluation,
        "score": score,
    }

    # Update the user's total score
    total_score_ref = db.collection("experiencePoints").document(uuid_user)
    total_score_doc = total_score_ref.get()

    # Adjust the score multiplier based on the difficulty level
    if level == "Beginner":
        score_multiplier = 100
    elif level == "Intermediate":
        score_multiplier = 300
    elif level == "Advanced":
        score_multiplier = 500
    else:
        score_multiplier = 1  # Default case, though it shouldn't reach here

    # Multiply the score before storing
    calculated_score = score * score_multiplier

    if total_score_doc.exists:
        total_score_data = total_score_doc.to_dict()
        new_total_score = total_score_data.get("score", 0) + calculated_score
    else:
        new_total_score = calculated_score

    total_score_ref.set({"score": new_total_score}, merge=True)
    

    new_question, new_topic = await generate_unique_question(level, session.get("askedTopics", []))
    
    session["currentQuestion"] = {"Topic": new_question}
    session["askedQuestions"].append(new_question)
    session["history"].append(history_entry)
    session["askedTopics"] = new_topic

    # question_ref = db.collection("Topics").document("AskedTopics")
    # question_ref.set({"Topics": firestore.ArrayUnion([{"Topic": new_topic}])}, merge=True)

    session_ref.set(session, merge=True)

    return {"evaluation": evaluation, "nextQuestion": new_question, "currentScore": session["score"]}

@app.get("/v1/progress/{sessionId}")
async def get_progress(sessionId: str, user_data: dict = Depends(get_firebase_user)):
    userId = user_data.get("uid")
    session_ref = db.collection("quiz_sessions").document(userId).collection("sessions").document(sessionId)
    session_doc = session_ref.get()
    
    if not session_doc.exists:
        raise HTTPException(status_code=400, detail="No active session found")
    
    session_data = session_doc.to_dict()
    return {"history": session_data.get("history", []), "score": session_data.get("score", 0)}

@app.post("/v1/chatwithimage")
async def chat(request: ChatRequestImage, user_data: dict = Depends(get_firebase_user)):
    chat_session_id = request.chatSessionId or str(uuid.uuid4())
    user_id = user_data.get("uid")

    history = await load_chat_history(user_id, chat_session_id)
    response = await send_message_to_gemini(request.message, request.imageUrl, history)

    new_history = history + [{"role": "user", "text": request.message, "image": request.imageUrl},
                             {"role": "model", "text": response}]
    await save_chat_history(user_id, chat_session_id, new_history)

    return {"reply": response, "chatSessionId": chat_session_id}

@app.post("/v1/login")
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

@app.post("/v1/refresh")
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

@app.get("/v1/protected-route")
def protected_route(user_data: dict = Depends(get_firebase_user)):
    return {"message": "Welcome to the protected route!", "user_data": user_data}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
