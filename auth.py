import logging
import os

import firebase_admin
import requests
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth, credentials, firestore

load_dotenv()

FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Define HTTPBearer for Authorization header
security = HTTPBearer()

def authenticate_with_firebase(email: str, password: str):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    data = {
        "email": email,
        "password": password,
        "returnSecureToken": True,
    }
    response = requests.post(url, json=data)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=response.json().get("error", {}).get("message", "Login failed"))
    return response.json()

# Helper to Refresh Token
def refresh_firebase_token(refresh_token: str):
    url = f"https://securetoken.googleapis.com/v1/token?key={FIREBASE_API_KEY}"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    response = requests.post(url, data=data)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Token refresh failed")
    return response.json()

def get_firebase_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get the user details from Firebase, based on TokenID"""
    id_token = credentials.credentials
    try:
        claims = auth.verify_id_token(id_token)
        return claims
    except Exception as e:
        logging.exception(e)
        raise HTTPException(status_code=401, detail='Unauthorized')


