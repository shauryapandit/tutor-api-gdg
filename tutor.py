from fastapi import FastAPI, HTTPException
import uvicorn
import os
import json
import pandas as pd
import requests
import json
from pydantic import BaseModel

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Load fintech topics from CSV
file_path = "./finance_topics_full.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# In-memory user session tracking
user_sessions = {}

class StartRequest(BaseModel):
    userId: str
    level: str

class AnswerRequest(BaseModel):
    userId: str
    answer: str

@app.post("/start")
def start_quiz(request: StartRequest):
    if request.level not in ["Beginner", "Intermediate", "Advanced"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty level")
    
    # Filter questions by difficulty
    questions = df[df["Difficulty"] == request.level].to_dict(orient="records")
    if not questions:
        raise HTTPException(status_code=500, detail="No questions found")
    
    # Store session
    user_sessions[request.userId] = {"level": request.level, "questions": questions, "history": []}
    
    # Select first question
    question = questions.pop(0)
    user_sessions[request.userId]["currentQuestion"] = question
    
    return {"message": f"Welcome! Here's your first question: {question['Topic']}"}
@app.post("/answer")
def answer_question(request: AnswerRequest):
    if request.userId not in user_sessions:
        raise HTTPException(status_code=400, detail="No active session. Start first!")

    session = user_sessions[request.userId]
    question = session["currentQuestion"]

    # Gemini API request payload
    data = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"Evaluate this answer for correctness: '{request.answer}' in relation to the topic '{question['Topic']}'. You are a financial education expert evaluating quiz answers. Provide a brief (2-3 sentence) evaluation stating correctness and a short correct explanation as COrrect Answer: ."}
                ]
            }
        ]
    }

    try:
        # Send request to Gemini API
        response = requests.post(
            GEMINI_URL,
            json=data,
            headers={"Content-Type": "application/json"}
        )

        # Debugging: Print raw response
        print("Raw Gemini Response:", response.text)

        # Parse response JSON
        response_data = response.json()

        # Extract evaluation text safely
        candidates = response_data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            evaluation = parts[0].get("text", "No response received.") if parts else "No content parts received."
        else:
            evaluation = "No valid response from Gemini."

        # Save history
        session["history"].append({"question": question["Topic"], "userAnswer": request.answer, "evaluation": evaluation})

        # Return next question or finish quiz
        if session["questions"]:
            session["currentQuestion"] = session["questions"].pop(0)
            return {"evaluation": evaluation, "nextQuestion": session["currentQuestion"]["Topic"]}
        else:
            del user_sessions[request.userId]  # End session
            return {"evaluation": evaluation, "message": "Quiz completed!"}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")


@app.get("/progress/{userId}")
def get_progress(userId: str):
    if userId not in user_sessions:
        raise HTTPException(status_code=400, detail="No active session found")
    return {"history": user_sessions[userId]["history"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)