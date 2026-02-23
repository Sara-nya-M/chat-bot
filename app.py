from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = Flask(__name__)

# -------------------------------------------------------
# Load Model (DialoGPT - free, runs on CPU)
# You can swap to "microsoft/DialoGPT-large" for better quality
# -------------------------------------------------------
print("Loading model... please wait (first run may take a few minutes)")
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

chat_history_ids = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history_ids
    user_message = request.json.get("message", "")

    # Encode user input
    new_input_ids = tokenizer.encode(
        user_message + tokenizer.eos_token,
        return_tensors="pt"
    )

    # Append to chat history (keep last 5 turns)
    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None
        else new_input_ids
    )

    # Limit context to 1000 tokens to avoid memory issues
    if bot_input_ids.shape[-1] > 1000:
        bot_input_ids = bot_input_ids[:, -1000:]

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )

    # Decode only the new response
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    return jsonify({"response": response})

@app.route("/reset", methods=["POST"])
def reset():
    global chat_history_ids
    chat_history_ids = None
    return jsonify({"status": "Chat reset successfully"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
