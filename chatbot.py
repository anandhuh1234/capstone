import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Llama3:
    def __init__(self, model_path):
        self.model_id = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map="auto").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("")]

    def generate_text(self, prompt, max_tokens=4096, temperature=0.6, top_p=0.9):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=self.tokenizer.eos_token_id,  # Use tokenizer's eos_token_id directly
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text


    def get_response(self, query, message_history=[], max_tokens=4096, temperature=0.6, top_p=0.9):
        user_prompt = message_history + [{"role": "user", "content": query}]
        prompt = self.tokenizer.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )
        response = self.generate_text(prompt, max_tokens, temperature, top_p)
        return response, user_prompt + [{"role": "assistant", "content": response}]

    def chatbot(self, system_instructions=""):
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
