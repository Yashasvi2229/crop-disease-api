from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_service import get_chat_response

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    language: str = "en"

@router.post("/chat", summary="Chat with AI Agricultural Expert")
async def chat(request: ChatRequest):
    """
    Chat with AI agricultural expert about farming questions.
    
    Args:
        question: User's farming question
        language: Language code (en, hi, ta, te)
    
    Returns:
        AI-generated answer with farming advice
    """
    try:
        answer = get_chat_response(request.question, request.language)
        return {
            "answer": answer,
            "language": request.language
        }
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error processing your question")
