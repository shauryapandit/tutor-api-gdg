from typing import Optional, Literal

from pydantic import BaseModel, EmailStr


class ChatRequestImage(BaseModel):
    # userId: str
    message: str
    chatSessionId: Optional[str] = None
    imageUrl: Optional[str] = None

class StartRequest(BaseModel):
    # userId: str
    level: Literal["Beginner", "Intermediate", "Advanced"]

class AnswerRequest(BaseModel):
    # userId: str
    sessionId: str
    answer: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RefreshRequest(BaseModel):
    refresh_token: str

class Portfolio(BaseModel):
    name: str
    age: int
    monthlyincome: float
    monthlysaving: float
    profession: Literal["Employed", "Self-Employed", "Unemployed"]
    primaryreasonforinvesting: str
    financialrisk: str
    expaboutinvesting: str
    estimateinvestingduration: int
    typesofinvestment: list[str]
    portfolio: list[str]