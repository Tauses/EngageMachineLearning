from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import torch

class ChatModelTrainer:
    def __init__(self, model_name="microsoft/DialoGPT-small", data_path="dialogues.jsonl"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.data_path = data_path

    def tokenize_function(self, examples):
        inputs = [f"User: {q} \nBot:" for q in examples["input"]]
        outputs = [r for r in examples["response"]]
        return self.tokenizer([i + o for i, o in zip(inputs, outputs)],
                              padding="max_length", truncation=True, max_length=128)

    def train(self):
        print("Indlæser data...")
        raw_dataset = load_dataset("json", data_files=self.data_path, split="train")

        print("Tokeniserer...")
        tokenized = raw_dataset.map(self.tokenize_function, batched=True)

        print("Træner model...")
        training_args = TrainingArguments(
            output_dir="./trained_chat_model",
            per_device_train_batch_size=2,
            num_train_epochs=3,
            logging_steps=10,
            save_strategy="no",
            weight_decay=0.01,
            warmup_steps=10
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        trainer.train()

        print("Gemmer model...")
        self.model.save_pretrained("trained_chat_model")
        self.tokenizer.save_pretrained("trained_chat_model")
        print("Model gemt.")

if __name__ == "__main__":
    trainer = ChatModelTrainer()
    trainer.train()
