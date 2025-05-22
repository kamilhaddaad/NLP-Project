from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class SummaryGenerator:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = PeftConfig.from_pretrained("summary_generator/TinyLlama_finetuned")
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(base_model, "summary_generator/TinyLlama_finetuned")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        self.model.to(self.device)
        
        self.bad_words = ["(show all)", "(hide all)", "click here", "More Books", "(less)"]
        self.bad_word_ids = [self.tokenizer.encode(bw, add_special_tokens=False) for bw in self.bad_words]

    def generate_summary(self, title:str, tone:str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot.",
            },
            {"role": "user", "content": f"Summarize the book {title} with a {tone} tone"},
        ]

        encoded = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        model_inputs = encoded.to(self.device)

        while True:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs, 
                    max_new_tokens=300,
                    num_beams=4,
                    early_stopping=True, 
                    repetition_penalty=1.2,
                    temperature=0.9, 
                    top_p=0.9, 
                    do_sample=True,
                    no_repeat_ngram_size=3,
                    bad_words_ids=self.bad_word_ids,
                    )

            decoded = self.tokenizer.batch_decode(generated_ids)
            answer = decoded[0]

            if "<|assistant|>\n" in answer:
                break

        answer_truncated = answer.split("<|assistant|>\n")[-1]

        if "</s>" in answer_truncated:
            answer_truncated = answer_truncated.split("</s>")[0]

        for word in self.bad_words:
            answer_truncated = answer_truncated.replace(word, "")

        return answer_truncated
