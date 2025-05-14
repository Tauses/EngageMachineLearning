from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

class MistralTrainer:
    def __init__(self, model_name, dataset_path):
        # Indlæs model og tokenizer
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.df = pd.read_csv(dataset_path)
        self.tokenized_data = self.prepare_data()

    def prepare_data(self):
        # Forbered data (antager allerede formateret som input-output par)
        dataset = Dataset.from_pandas(self.df[['formatted_input', 'formatted_output']])

        def tokenize_function(examples):
            prompts = [f"{inp} {out}" for inp, out in zip(examples['formatted_input'], examples['formatted_output'])]
            return self.tokenizer(prompts, truncation=True, padding="max_length", max_length=512)

    def train(self):
        # Træningsparametre
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_data,
            eval_dataset=self.tokenized_data
        )

        trainer.train()

    def save_model(self, save_path):
        # Gem den trænede model
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


# Eksempel på brug:
trainer = MistralTrainer(model_name="mistralai/Mistral-7B-Instruct-v0.1", dataset_path="formatted_data.csv")
trainer.train()
trainer.save_model("./fine_tuned_mistral")
