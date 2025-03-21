from fastapi import FastAPI, HTTPException
import uvicorn
import os
import uuid 
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

# Gemini API setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env!")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

SYSTEM_PROMPT = """
You are a financial education expert. Based on the user's selected difficulty level, generate a question from the following topics:

1. If the user selects 'Beginner', ask simple fundamental financial concepts.
2. If the user selects 'Intermediate', ask about financial instruments and market trends.
3. If the user selects 'Advanced', ask about technical analysis and risk management.

Generate a relevant question from the provided list of topics. If no matching topic is found, create a relevant financial question.

**Instructions:**
- Only return the question, nothing else.
- Do NOT include greetings, explanations, or any additional sentences.
- Ensure the question is concise, clear, and relevant to the difficulty level.

"""

class StartRequest(BaseModel):
    userId: str
    level: str

class AnswerRequest(BaseModel):
    userId: str
    sessionId: str
    answer: str

@app.post("/start")
async def start_quiz(request: StartRequest):
    if request.level not in ["Beginner", "Intermediate", "Advanced"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty level")

    # Generate a unique session ID
    session_id = str(uuid.uuid4())

    # Generate a question dynamically
    prompt_text = f"""
    {SYSTEM_PROMPT}
    User selected difficulty level: {request.level}
    Provide a relevant and short question based on the difficulty level.
    """

    try:
        response = model.generate_content(prompt_text)
        generated_question = response.text if response.text else "No question generated."
    except Exception as e:
        print(f"Error generating question with Gemini API: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate question.")

    session_data = {
        "userId": request.userId,
        "sessionId": session_id,
        "level": request.level,
        "history": [],
        "currentQuestion": {"Topic": generated_question},
    }

    # Store session data in Firestore under quiz_sessions/{userId}/sessions/{sessionId}
    db.collection("quiz_sessions").document(request.userId).collection("sessions").document(session_id).set(session_data)

    return {"sessionId": session_id, "message": f"Here's your first question: {generated_question}"}

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
    session_ref = db.collection("quiz_sessions").document(request.userId).collection("sessions").document(request.sessionId)
    session_doc = session_ref.get()

    if not session_doc.exists:
        raise HTTPException(status_code=400, detail="No active session found!")

    session = session_doc.to_dict()
    question = session["currentQuestion"]

    # Evaluate the user's answer
    evaluation = await send_to_gemini(request.answer, question["Topic"])

    history_entry = {
        "question": question["Topic"],
        "userAnswer": request.answer,
        "evaluation": evaluation,
    }
    session["history"].append(history_entry)

    # Generate a new question
    prompt_text = f"""
    {SYSTEM_PROMPT}
    The previous question was '{question["Topic"]}'.
    Now generate a new financial question.
    
    Ensure the question is short and relevant to the topic.
    """

    try:
        response = model.generate_content(prompt_text)
        new_question = response.text if response.text else "No question generated."
    except Exception as e:
        print(f"Error generating next question with Gemini API: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate next question.")

    session["currentQuestion"] = {"Topic": new_question}

    # Save updated session to Firestore
    session_ref.set(session, merge=True)

    return {"evaluation": evaluation, "nextQuestion": new_question}

@app.get("/progress/{userId}/{sessionId}")
def get_progress(userId: str, sessionId: str):
    session_ref = db.collection("quiz_sessions").document(userId).collection("sessions").document(sessionId)
    session_doc = session_ref.get()

    if not session_doc.exists:
        raise HTTPException(status_code=400, detail="No active session found")
    
    return {"history": session_doc.to_dict().get("history", [])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
