import os
import re
import torch
from threading import Thread
from typing import Dict, List
from transformers import TextIteratorStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


MAX_LENGTH = 128
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 10
TOP_P = 0.95
TOP_K = 40
REPETITION_PENALTY = 1.0
NO_REPEAT_NGRAM_SIZE = 0
DO_SAMPLE = True
DEFAULT_STREAM = True

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.tokenizer = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def preprocess(self, request: dict):
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token) if self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token) else self.tokenizer.eos_token_id,
        ]
        generate_args = {
            "max_length": request.get("max_tokens", MAX_LENGTH),
            "temperature": request.get("temperature", TEMPERATURE),
            "top_p": request.get("top_p", TOP_P),
            "top_k": request.get("top_k", TOP_K),
            "repetition_penalty": request.get("repetition_penalty", REPETITION_PENALTY),
            "no_repeat_ngram_size": request.get("no_repeat_ngram_size", NO_REPEAT_NGRAM_SIZE),
            "do_sample": request.get("do_sample", DO_SAMPLE),
            "use_cache": True,
            "eos_token_id": terminators,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": MAX_NEW_TOKENS
        }
        request["generate_args"] = generate_args
        return request

    def stream(self, input_ids: list, generation_args: dict):
        streamer = TextIteratorStreamer(self.tokenizer)
        generation_config = GenerationConfig(**generation_args)
        generation_kwargs = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": generation_args["max_new_tokens"],
            "streamer": streamer,
        }

        with torch.no_grad():
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            def inner():
                for text in streamer:
                    yield text
                thread.join()

        return inner()

    def predict(self, request: Dict):
        messages = request.pop("messages")
        stream = request.pop("stream", DEFAULT_STREAM)
        generation_args = request.pop("generate_args")

        model_inputs = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        inputs = self.tokenizer(model_inputs, return_tensors="pt")
        inputs = self.tokenizer(model_inputs, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs["input_ids"].to("cuda")

        if stream:
            return self.stream(input_ids, generation_args)

        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, **generation_args)
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"output": output_text}

print("Loading model")
model = Model()
model.load()


def inference(input_text):
    request = {
        "messages": [{"role": "user", "content": input_text}],
        "max_tokens": 128,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "do_sample": True,
        "stream": False,
    }


    request = model.preprocess(request)
    response = model.predict(request)

    generated_answer = response["output"].strip()

    return generated_answer

# Starting inference
print("model inference\n")
inference(input_text="what is 1+1?")
