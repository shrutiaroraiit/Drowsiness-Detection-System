from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import List

from utils.state_detector import get_drowsy_state
from api.llm_handler import get_local_llm_response

router = APIRouter()

# Keep conversation history globally for simplicity
chat_history = None

# Input schema for /predict route
class FeatureInput(BaseModel):
    vit_features: List[List[float]]
    traditional_features: List[List[float]]

# Drowsiness state prediction
@router.post("/predict")
async def predict_state(data: FeatureInput):
    state = get_drowsy_state(data.vit_features, data.traditional_features)

    if state == "Drowsy":
        return {
            "state": state,
            "message": "Is there a coffee shop nearby?"
        }
    else:
        return {
            "state": state,
            "message": "You're alert and active!"
        }

# Chat route with LLM
@router.post("/chat")
async def chat_with_user(message: str = Body(...)):
    global chat_history

    # Handle stop signal to reset conversation
    if message.lower().strip() == "stop":
        chat_history = None
        return {"message": "Rechecking your state... Please send new features to /predict."}

    # Get LLM response and update history
    response, chat_history = get_local_llm_response(message, chat_history)
    return {"message": response}
