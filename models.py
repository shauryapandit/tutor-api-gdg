from pydantic import BaseModel


class ChatRequest(BaseModel):
    # userId: str
    message: str
    chatSessionId: str
    
class StartRequest(BaseModel):
    userId: str
    level: str

class AnswerRequest(BaseModel):
    userId: str
    sessionId: str
    answer: str

class LoginRequest(BaseModel):
    email: str
    password: str

class RefreshRequest(BaseModel):
    refresh_token: str