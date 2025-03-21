from fastapi import FastAPI, HTTPException
import uvicorn
import os
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = FastAPI()

# Load Firebase credentials
firebase_creds_path = "./serviceAccountKey.json"
if not os.path.exists(firebase_creds_path):
    raise RuntimeError("Firebase credentials file missing!")

cred = credentials.Certificate(firebase_creds_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load fintech topics
file_path = "./finance_topics_full.csv"
if not os.path.exists(file_path):
    raise RuntimeError("CSV file missing!")

df = pd.read_csv(file_path)

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env!")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# System Prompt for Dynamic Question Generation
SYSTEM_PROMPT = """
You are a financial education expert. Based on the user's selected difficulty level, generate a question from the following topics:

1. If the user selects 'Beginner', ask simple fundamental financial concepts.
2. If the user selects 'Intermediate', ask about financial instruments and market trends.
3. If the user selects 'Advanced', ask about technical analysis and risk management.

Generate a relevant question from the provided list of topics. If no matching topic is found, create a relevant financial question.
"""

# In-memory user session tracking
user_sessions = {}

class StartRequest(BaseModel):
    userId: str
    level: str

class AnswerRequest(BaseModel):
    userId: str
    answer: str

@app.post("/start")
async def start_quiz(request: StartRequest):
    if request.level not in ["Beginner", "Intermediate", "Advanced"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty level")
    
    questions = df[df["Difficulty"] == request.level].to_dict(orient="records")
    if not questions:
        raise HTTPException(status_code=500, detail="No questions found")
    
    # Generate a question dynamically if no relevant question is available in the dataset
    prompt_text = f"""
    {SYSTEM_PROMPT}
    User selected difficulty level: {request.level}
    Available topics: {', '.join(df['Topic'].unique())}
    
    Provide a relevant and short question based on the difficulty level.
    Question must not be more than one small sentence
    """
    
    try:
        response = model.generate_content(prompt_text)
        generated_question = response.text if response.text else "No question generated."
    except Exception as e:
        print(f"Error generating question with Gemini API: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate question.")
    
    user_sessions[request.userId] = {"level": request.level, "questions": questions, "history": [], "currentQuestion": {"Topic": generated_question}}
    
    return {"message": f"Welcome! Here's your first question: {generated_question}"}

async def send_to_gemini(user_answer: str, question_topic: str) -> str:
    """Sends the user response and question to Gemini for evaluation."""
    prompt_text = f"""
    You are evaluating a financial quiz answer.
    
    **Question:** {question_topic}
    **User's Answer:** {user_answer}
    
    Provide feedback on correctness and a brief explanation.
    """

    try:
        response = model.generate_content(prompt_text)
        return response.text if response.text else "No response received."
    except Exception as e:
        print(f"Error communicating with Gemini API: {e}")
        raise HTTPException(status_code=500, detail="Failed to process request.")

@app.post("/answer")
async def answer_question(request: AnswerRequest):
    if request.userId not in user_sessions:
        raise HTTPException(status_code=400, detail="No active session. Start first!")

    session = user_sessions[request.userId]
    question = session["currentQuestion"]

    evaluation = await send_to_gemini(request.answer, question["Topic"])

    history_entry = {
        "question": question["Topic"],
        "userAnswer": request.answer,
        "evaluation": evaluation,
    }
    session["history"].append(history_entry)

    # Update Firestore
    db.collection("quiz_sessions").document(request.userId).set({
        "userId": request.userId,
        "history": session["history"]
    }, merge=True)

    if session["questions"]:
        session["currentQuestion"] = session["questions"].pop(0)
        return {"evaluation": evaluation, "nextQuestion": session["currentQuestion"]["Topic"]}
    else:
        del user_sessions[request.userId]
        return {"evaluation": evaluation, "message": "Quiz completed!"}

@app.get("/progress/{userId}")
def get_progress(userId: str):
    doc = db.collection("quiz_sessions").document(userId).get()
    if not doc.exists:
        raise HTTPException(status_code=400, detail="No active session found")
    
    return {"history": doc.to_dict().get("history", [])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
