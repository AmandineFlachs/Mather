import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, setup_chat_format
from peft import LoraConfig
from datasets import load_dataset, concatenate_datasets
from accelerate import PartialState

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

def convert_to_conversational_format(x, input, output):
    """Convert a pair of input/output (typically question/answer) to conversational format."""

    return {"messages": [{"role": "user", "content": x[input]}, {"role": "assistant", "content": x[output]}]}

def create_trainer():
    """Return a trainer configured to generate a low-rank adapter (LoRA) for Mistral-7B-Instruct-v0.3 on the MathInstruct, MetaMathQA and GSM8K datasets."""

    datasets = [
        load_dataset("TIGER-Lab/MathInstruct", split="train"), # format: instruction, output
        load_dataset("meta-math/MetaMathQA", split="train"), # format: query, response
        load_dataset("openai/gsm8k", "main", split="train"), # format: question, answer
    ]

    offset = 0
    datasets[offset+0] = datasets[offset+0].map(lambda x: convert_to_conversational_format(x, "instruction", "output"), remove_columns=['source', 'output', 'instruction'])
    datasets[offset+1] = datasets[offset+1].map(lambda x: convert_to_conversational_format(x, "query", "response"), remove_columns=['type', 'query', 'original_question', 'response'])
    datasets[offset+2] = datasets[offset+2].map(lambda x: convert_to_conversational_format(x, "question", "answer"), remove_columns=['question', 'answer'])

    dataset = concatenate_datasets(datasets)

    device_string = PartialState().process_index

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={'':device_string},
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model, tokenizer = setup_chat_format(model, tokenizer)

    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=200000,
        logging_steps=1000,
        learning_rate=1e-6,
        weight_decay=0,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=False,
        lr_scheduler_type="constant",
        report_to="tensorboard",
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=32,
        bias="none",
        task_type="CAUSAL_LM"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=512,
    )

    return trainer

def save_adapter(trainer):
    """Save adapter to adapter/adapter_config.json and adapter/adapter_model.safetensors."""

    model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    model.save_pretrained("./adapter")

def main():
    """Generate a Mistral-7B-Instruct-v0.3 fine-tune (more precisely, an adapter) and save it."""

    trainer = create_trainer()

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()

    save_adapter(trainer)

if __name__ == "__main__":
    main()
