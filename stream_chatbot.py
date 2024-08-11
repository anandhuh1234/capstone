import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Llama3:
    def __init__(self, model_path):
        self.model_id = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map="auto").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("")]

    def generate_text(self, prompt, max_tokens=2048, temperature=0, top_p=0.9, chunk_size=256):
        # ... (rest of the function)
        for _ in range(0, max_tokens, chunk_size):
            output = self.model.generate(
                # ... (rest of the generation arguments)
            )
            generated_chunk = output[0].tolist()
            actual_chunk_size = len(generated_chunk)
    
            # Handle incomplete chunk (pad if necessary)
            if actual_chunk_size < chunk_size:
                generated_chunk += [self.tokenizer.pad_token_id] * (chunk_size - actual_chunk_size)
    
            print(self.tokenizer.decode(generated_chunk, skip_special_tokens=True))
            # Update input_ids with generated tokens
            input_ids = input_ids[:, 1:] + torch.tensor(generated_chunk, device=self.device)
        return self.tokenizer.decode(generated, skip_special_tokens=True


    def get_response(self, query, message_history=[], max_tokens=4096, temperature=0.1, top_p=0.9):
        user_prompt = message_history + [{"role": "user", "content": query}]
        prompt = self.tokenizer.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )
        response = self.generate_text(prompt, max_tokens, temperature, top_p, chunk_size=256)
        return response, user_prompt + [{"role": "assistant", "content": response}]


    def chatbot(self, system_instructions="You are a helpful assistant."):
        conversation = [{"role": "system", "content": system_instructions}]
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the chatbot. Goodbye!")
                break
            response, conversation = self.get_response(user_input, conversation)
            print(f"Assistant: {response}")

if __name__ == "__main__":
    bot = Llama3("meta-llama/Meta-Llama-3.1-8B-Instruct")
    bot.chatbot()
