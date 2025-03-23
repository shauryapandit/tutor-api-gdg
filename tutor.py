import os
import uuid
import firebase_admin
import google.generativeai as genai
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from firebase_admin import credentials, firestore
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

# Load Firebase credentials
firebase_creds_path = "./serviceAccountKey.json"
if not os.path.exists(firebase_creds_path):
    raise RuntimeError("Firebase credentials file missing!")

cred = credentials.Certificate(firebase_creds_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env!")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

SYSTEM_PROMPT = """
You are a financial education expert. Based on the user's selected difficulty level, generate a unique question from the following topics:

1. If the user selects 'Beginner', ask simple fundamental financial concepts.
2. If the user selects 'Intermediate', ask about financial instruments and market trends.
3. If the user selects 'Advanced', ask about technical analysis and risk management.

**Instructions:**
- Do NOT repeat any previously asked questions.
- Return only the question, without explanations or greetings.
- Ensure the question is concise, clear, and relevant to the difficulty level.
"""

SCORE_PROMPT = """
You are evaluating a financial quiz answer based on accuracy.

**Scoring Criteria:**
- Beginner: 1 point for correct, 1 point for partial, 0 for incorrect.
- Intermediate: 2 points for correct, 1 point for partial, 0 for incorrect.
- Advanced: 3 points for correct, 2 points for partial, 0 for incorrect.

**Instructions:**
- Provide a score (0, 1, 2, or 3) based on the difficulty level.
- Explain briefly why the score was assigned.

**Format:**
Score: X
Explanation: Y
Reason: (Keep it Brief)
"""

class StartRequest(BaseModel):
    userId: str
    level: str

class AnswerRequest(BaseModel):
    userId: str
    sessionId: str
    answer: str
    session_id: str

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
    session_data["history"].append(history_entry)

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
