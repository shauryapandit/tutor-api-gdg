# FastAPI Financial Quiz and Chat API

This FastAPI-based application provides a financial quiz system and an AI-powered chat interface for users to interact with. It integrates Firebase authentication and Gemini AI for enhanced functionality.

## Features
- User authentication via Firebase
- Start and manage financial quiz sessions
- Evaluate user responses and track progress
- AI-powered chat with image support
- Secure token authentication and refresh functionality

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables in a `.env` file:
   ```env
   FIREBASE_CREDENTIALS_PATH=path/to/firebase/credentials.json
   ```
5. Run the application:
   ```bash
   uv run main.py
   ```

## API Endpoints

### Root Endpoint
```http
GET /
```
**Response:**
```json
{
  "Status": "Active"
}
```

### Start Quiz Session
```http
POST /v1/start
```
**Request Body:**
```json
{
  "level": "Beginner"
}
```
**Response:**
```json
{
  "sessionId": "unique-session-id",
  "message": "First question text"
}
```

### Submit Answer
```http
POST /v1/answer
```
**Request Body:**
```json
{
  "sessionId": "unique-session-id",
  "answer": "User's answer"
}
```
**Response:**
```json
{
  "evaluation": "AI-generated feedback",
  "nextQuestion": "Next question text",
  "currentScore": 10
}
```

### Get Quiz Progress
```http
GET /v1/progress/{sessionId}
```
**Response:**
```json
{
  "history": [
    {
      "question": "Previous question",
      "userAnswer": "User's response",
      "evaluation": "AI feedback",
      "score": 10
    }
  ],
  "score": 20
}
```

### Chat with Image
```http
POST /v1/chatwithimage
```
**Request Body:**
```json
{
  "chatSessionId": "session-id",
  "message": "User's message",
  "imageUrl": "https://example.com/image.jpg" (Optional)
}
```
**Response:**
```json
{
  "reply": "AI-generated response",
  "chatSessionId": "session-id"
}
```

### User Authentication
#### Login
```http
POST /v1/login
```
**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```
**Response:**
```json
{
  "id_token": "auth-token",
  "refresh_token": "refresh-token",
  "expires_in": 3600
}
```
#### Refresh Token
```http
POST /v1/refresh
```
**Request Body:**
```json
{
  "refresh_token": "refresh-token"
}
```
**Response:**
```json
{
  "id_token": "new-auth-token",
  "refresh_token": "new-refresh-token",
  "expires_in": 3600
}
```

### Protected Route
```http
GET /v1/protected-route
```
**Response:**
```json
{
  "message": "Welcome to the protected route!",
  "user_data": {
    "uid": "user-id"
  }
}
```

## Error Handling
The API returns appropriate HTTP status codes and error messages:
- `400 Bad Request` for invalid inputs
- `401 Unauthorized` for authentication failures
- `500 Internal Server Error` for unexpected failures

## Deployment
To deploy using Docker:
```bash
docker build -t fastapi-financial-quiz .
docker run -p 8000:8000 fastapi-financial-quiz
```

## License
This project is licensed under the MIT License.