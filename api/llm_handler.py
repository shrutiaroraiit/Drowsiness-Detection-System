from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer at startup
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Store chat history globally (or per-session if needed)
chat_history_ids = None

def get_local_llm_response(user_input, history=None):
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Concatenate new input to history
    bot_input_ids = torch.cat([history, new_input_ids], dim=-1) if history is not None else new_input_ids

    chat_history = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    response_text = tokenizer.decode(chat_history[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response_text, chat_history
