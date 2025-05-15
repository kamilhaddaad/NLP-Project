import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

# Load data
csv_path = 'data/book_genre_prediction.csv'
df = pd.read_csv(csv_path)

# Create prompt + target
PROMPT_TEMPLATE = 'Title: {title}. Genre: {genre}. Description:'
def make_prompt(row):
    return PROMPT_TEMPLATE.format(title=row['title'], genre=row['genre'])

def make_target(row):
    return row['summary'] if isinstance(row['summary'], str) else ''

df = df.dropna(subset=['title', 'genre', 'summary'])
df['text'] = df.apply(lambda row: make_prompt(row) + ' ' + make_target(row), axis=1)

# Create Huggingface Dataset
hf_dataset = Dataset.from_pandas(df[['text']])

# Load tokenizer and modell
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenizing
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Arguments for training
training_args = TrainingArguments(
    output_dir='./gpt2_blurb_finetuned',
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=5000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save Model
trainer.save_model('./gpt2_blurb_finetuned')
tokenizer.save_pretrained('./gpt2_blurb_finetuned') 