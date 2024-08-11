import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# v04
class Llama3:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

    def generate_text_stream(self, prompt, max_tokens=2048, temperature=0.7, top_p=0.9):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)

        generated_ids = input_ids

        for _ in range(max_tokens):
            output = self.model.generate(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,  # Generate one token at a time
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )
            
            next_token_id = output.sequences[:, -1:]
            generated_ids = torch.cat((generated_ids, next_token_id), dim=-1)
            
            # Update the attention mask to account for the new token
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=-1)
            
            next_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            yield next_token
            
            if next_token_id in self.tokenizer.all_special_ids:  # Stop if EOS or other special token is generated
                break

    def get_response_stream(self, query, message_history=[], max_tokens=2048, temperature=0.7, top_p=0.9):
        user_prompt = message_history + [{"role": "user", "content": query}]
        prompt = self.tokenizer.apply_chat_template(user_prompt, tokenize=False, add_generation_prompt=True)
        return self.generate_text_stream(prompt, max_tokens, temperature, top_p), user_prompt

    def chatbot(self, system_instructions="You are a helpful assistant."):
        conversation = [{"role": "system", "content": system_instructions}]
        while True:
            user_input = input("User: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the chatbot. Goodbye!")
                break

            response_stream, conversation = self.get_response_stream(user_input, conversation)
            print("Assistant: ", end="", flush=True)

            for token in response_stream:
                print(token, end="", flush=True)
            print()

if __name__ == "__main__":
    bot = Llama3("meta-llama/Meta-Llama-3.1-8B-Instruct")
    bot.chatbot()
