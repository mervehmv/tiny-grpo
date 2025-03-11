from collections.abc import Callable
import json
from pathlib import Path
import random
import re
import os
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
    AutoModelForCausalLM
)
from datasets import load_dataset

def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


@torch.no_grad()
def rollout(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task,
        },
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to("cuda")
    


    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # 2. sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. determine rewards
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    response_lengths = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        # search answer tag
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        answer = answer_match.group(1) if answer_match else None


        reward = 0
        if answer is not None:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.01

        returns[i] = reward
        response_lengths[i] = len(completion)

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions, response_lengths, answer


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)



def main():
    seed = 42
    device_index = 0
    train_batch_size = 16
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 5 #12
    rollouts_per_step = 16 #32
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 1024
    top_p = 1.0
    temperature = 1.0

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    checkpoint_dir = Path("/content/drive/MyDrive/tiny-grpo-gsmk8/output2/step_899")
    model, tokenizer = load_model(checkpoint_dir, device_map=device)

    # Math500 DATASET
    dataset = load_dataset("ricdomolm/MATH-500")
    test_data = dataset["test"]

    prompt_loader = DataLoader(
        test_data,
        batch_size=rollouts_per_step,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
    )


    with open("./evaluate.txt", "a", encoding="utf-8") as file:
        for k, prompt_batch in enumerate(prompt_loader):

            questions = prompt_batch["problem"]
            answers = prompt_batch["answer"]

            with torch.no_grad():
                for q, a in zip(questions, answers):

                    sequence_ids, returns, action_mask, completions, response_lengths, ans = rollout(
                        model,
                        tokenizer,
                        q,
                        a,
                        num_rollouts=group_size,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    print(
                        f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, sequence_ids={sequence_ids.shape}"
                    )


                    file.write(f"Question\n{q}\nThink\n{completions[0]}\nAnswer\n{ans}\nOracle Answer\n{a}\n\n")

                    os.sync()

                    
            torch.cuda.empty_cache()
            

if __name__ == "__main__":
    main()
