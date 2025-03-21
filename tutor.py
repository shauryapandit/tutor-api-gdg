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

Do not greet the user. Directly ask the question.
"""


class StartRequest(BaseModel):
    userId: str
    level: str

class AnswerRequest(BaseModel):
    userId: str
    answer: str
    session_id: str

@app.post("/start")
async def start_quiz(request: StartRequest):

    if request.level not in ["Beginner", "Intermediate", "Advanced"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty level")
    
    questions = df[df["Difficulty"] == request.level].to_dict(orient="records")
    if not questions:
        raise HTTPException(status_code=500, detail="No questions found")

    session_id = uuid.uuid4().hex
    
    # Generate a question dynamically if no relevant question is available in the dataset
    prompt_text = f"""
    {SYSTEM_PROMPT}
    User selected difficulty level: {request.level}
    Available topics: {', '.join(df['Topic'].unique())}
    
    Provide a relevant and short question based on the difficulty level.
    Question must not be more than one small sentence.
    """
    
    try:
        response = model.generate_content(prompt_text)
        generated_question = response.text if response.text else "No question generated."
    except Exception as e:
        print(f"Error generating question with Gemini API: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate question.")
    
    doc_ref = db.collection("quiz_sessions").document(request.userId).collection("sessions").document(session_id)
    doc_ref.set({
        "level": request.level,
        "questions": questions,
        "history": [],
        "currentQuestion": {"Topic": generated_question}
    })
    return {"message": f"{generated_question}",
            "session_id": session_id}

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
    # if request.userId not in user_sessions:
    #     raise HTTPException(status_code=400, detail="No active session. Start first!")
    session_id = request.session_id
    doc_ref = db.collection("quiz_sessions").document(request.userId).collection("sessions").document(session_id)

    # checking if the `session_id` exists in the Firestore database
    if not doc_ref.get().exists:
        raise HTTPException(status_code=400, detail="No active session found")
    
    session_data = doc_ref.get().to_dict()
    question = session_data.get("currentQuestion")
    if question is None:
        raise HTTPException(status_code=400, detail="No current question found")
    
    if not request.answer:
        raise HTTPException(status_code=400, detail="Answer cannot be empty")

    # Evaluate the user's answer
    evaluation = await send_to_gemini(request.answer, question["Topic"])

    history_entry = {
        "question": question["Topic"],
        "userAnswer": request.answer,
        "evaluation": evaluation,
    }
    session_data["history"].append(history_entry)

    # Save progress to Firestore
    doc_ref.set(session_data, merge=True)


    # If there are more questions, generate a new question using Gemini
    if session_data["questions"]:
        next_topic = session_data["questions"].pop(0)["Topic"]  # Get the next topic
        prompt_text = f"""
        {SYSTEM_PROMPT}
        The previous question was '{question["Topic"]}'.
        Now generate a new financial question based on the next topic: {next_topic}.
        
        Ensure the question is short and relevant to the topic.
        """

        try:
            response = model.generate_content(prompt_text)
            new_question = response.text if response.text else "No question generated."
            session_data["currentQuestion"] = {"Topic": new_question}
            doc_ref.set(session_data, merge=True)
            return {"evaluation": evaluation, "nextQuestion": new_question}
        except Exception as e:
            print(f"Error generating next question with Gemini API: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate next question.")
    else:
        return {"evaluation": evaluation, "message": "Quiz completed!"}


# @app.get("/progress/{userId}")
# def get_progress(userId: str):
#     doc = db.collection("quiz_sessions").document(userId).get()
#     if not doc.exists:
#         raise HTTPException(status_code=400, detail="No active session found")
    
#     return {"history": doc.to_dict().get("history", [])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
