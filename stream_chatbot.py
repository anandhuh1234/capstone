import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Llama3:
    def __init__(self, model_path):
        self.model_id = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("")]

    def generate_text_stream(self, prompt, max_tokens=2048, temperature=0, top_p=0.9):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            eos_token_id=self.tokenizer.eos_token_id,  # Use tokenizer's eos_token_id directly
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_scores=True
        )

        # Stream the response token by token
        generated_text = ""
        for token_id in output.sequences[0]:
            token = self.tokenizer.decode(token_id, skip_special_tokens=True)
            generated_text += token + " "
            yield token

    def get_response_stream(self, query, message_history=[], max_tokens=4096, temperature=0.1, top_p=0.9):
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
            print("Assistant: ", end="")

            for token in response_stream:
                print(token, end="", flush=True)
            print()

if __name__ == "__main__":
    bot = Llama3("meta-llama/Meta-Llama-3.1-8B-Instruct")
    bot.chatbot()
