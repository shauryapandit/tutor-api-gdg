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
    GEMINI_API_KEY=your_gemini_api_key
    FIREBASE_API_KEY=your_firebase_api_key
    FIREBASE_SERVICE_ACCOUNT_KEY=your_base64_encoded_service_account_json
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
  "message": "Issuer of the token, indicating Firebase authentication (e.g., https://securetoken.google.com/example-project-72981).",
  "user_data": {
    "iss": "Issuer of the token, indicating Firebase authentication",
    "aud": " Audience, which is your Firebase project identifier.",
    "auth_time": "Timestamp (UNIX format) when the user was authenticated.",
    "user_id": "Unique Firebase User ID",
    "sub": "Unique Firebase User ID",
    "iat": "Issued At timestamp, indicating when the token was generated.",
    "exp": "Expiry timestamp, after which the token is invalid.",
    "email": "User’s verified email address.",
    "email_verified": "Boolean indicating whether the email is verified (true or false).",
    "firebase": {
      "identities": {
        "email": [
          "User’s verified email address."
        ]
      },
      "sign_in_provider": "password"
    },
    "uid": "Unique Firebase User ID"
  }
}
```


## Authentication
This API uses Bearer token authentication. Include the `Authorization: Bearer YOUR_ACCESS_TOKEN` header in each request.
execept for the following endpoints


### Endpoints That Do Not Require Authentication
The following endpoints do not require a Bearer token in the request header:
- `GET /`
- `POST /v1/login`
- `POST /v1/refresh`

### Example cURL Request:
```bash
curl -X 'POST' \
  'https://tutor-api-gdg.vercel.app/v1/chatwithimage' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer YOUR_ACCESS_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
  "message": "string",
  "chatSessionId": "hey"
}'
```

## Using the API in Next.js
To interact with the API in a Next.js application, you can use the `fetch` API or Axios.

### Example Using fetch:
```javascript
async function chatWithImage() {
  const response = await fetch('https://tutor-api-gdg.vercel.app/v1/chatwithimage', {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
      'Authorization': `Bearer YOUR_ACCESS_TOKEN`,
    },
    body: JSON.stringify({
      message: 'Hello',
      chatSessionId: 'hey',
    })
  });

  const data = await response.json();
  console.log(data);
}

chatWithImage();
```

### Example Using Axios:
```javascript
import axios from 'axios';

async function chatWithImage() {
  try {
    const response = await axios.post(
      'https://tutor-api-gdg.vercel.app/v1/chatwithimage',
      {
        message: 'Hello',
        chatSessionId: 'hey',
      },
      {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          'Authorization': `Bearer YOUR_ACCESS_TOKEN`,
        },
      }
    );

    console.log(response.data);
  } catch (error) {
    console.error(error);
  }
}

chatWithImage();
```

This demonstrates how to interact with the FastAPI endpoints using authentication in a Next.js application.