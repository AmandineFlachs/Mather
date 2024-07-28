import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
from peft import PeftModel

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

def main():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model, tokenizer = setup_chat_format(model, tokenizer)

    adapter = PeftModel.from_pretrained(model, "./adapter", torch_dtype=torch.bfloat16)

    model = adapter.merge_and_unload(progressbar=True)
    model.save_pretrained("./merged"),

if __name__ == "__main__":
    main()
