from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("trained_chat_model")
model = AutoModelForCausalLM.from_pretrained("trained_chat_model")

def chat():
    print("🤖 Trained ChatBot: Hi! Let’s talk about games. Type 'bye' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["bye", "exit", "quit"]:
            print("ChatBot: See you next time 🎮")
            break

        new_input = f"User: {user_input}\nBot:"
        input_ids = tokenizer.encode(new_input, return_tensors="pt")

        output = model.generate(
            input_ids,
            max_length=150,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )

        reply = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        print("ChatBot:", reply)

if __name__ == "__main__":
    chat()
