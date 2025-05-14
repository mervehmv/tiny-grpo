import os
import json
import torch
import random
import re
from pathlib import Path
from typing import Any, Iterator, Optional
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    GenerationConfig,
)
import wandb
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch
from IPython.display import display

# Define system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""

def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)

# Load the Qwen2-VL-7B-Instruct model
def load_model(model_name_or_path: str, device_map=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    return model, tokenizer, processor

# Compute log probabilities from logits
def sequence_log_probs_from_logits(logits: torch.Tensor, output_ids: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)

# Compute log probabilities for sequences
def sequences_log_probs(model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs

def collate_fn(batch):
    answers = []
    for item in batch:
        if item["options"] != []:
            if item["answer"] == "A":
                item["answer"] = item["options"][0]
                answers.append(item["options"][0])
            elif item["answer"] == "B":
                item["answer"] = item["options"][1]
                answers.append(item["options"][1])
            elif item["answer"] == "C":
                item["answer"] = item["options"][2]
                answers.append(item["options"][2])
            elif item["answer"] == "D":
                item["answer"] = item["options"][3]
                answers.append(item["options"][3])
            elif item["answer"] == "E":
                item["answer"] = item["options"][4]
                answers.append(item["options"][4])
        else:
            answers.append(item["answer"])

    return {
        "question": [item["question"] for item in batch],
        "answer": answers,
        "decoded_image": [item["decoded_image"] for item in batch],  # keep list of PIL images
    }

# Rollout function to generate responses and compute rewards
@torch.no_grad()
def rollout(
    model,
    tokenizer,
    processor,
    image: Image.Image,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    model.eval()
    chat_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # tells tokenizer to insert <|image|>
                {"type": "text", "text": task},  # your question/task string
            ]
        }
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    #chat_prompt = "<|image|>\n" + chat_prompt  # <-- add this line


    processed_inputs = processor(
        text=chat_prompt,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    # Move tensors to CUDA **after** extracting grid_thw
    image_tensor = processed_inputs["pixel_values"]
    input_ids = processed_inputs["input_ids"]
    attention_mask = processed_inputs["attention_mask"]
    image_grid_thw = processed_inputs.get("image_grid_thw")

    # Repeat tensors for rollouts
    input_ids = input_ids.repeat(num_rollouts, 1).to("cuda")
    attention_mask = attention_mask.repeat(num_rollouts, 1).to("cuda")
    image_tensor = image_tensor.repeat(num_rollouts, 1, 1, 1).to("cuda")
    # If shape is [1, 3], repeat it to [B, 3]
    image_grid_thw = image_grid_thw.expand(num_rollouts, -1).to("cuda")

    
    # Generate responses
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Generate with image + grid info
    sequence_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=image_tensor,
        image_grid_thw=image_grid_thw,
        generation_config=generation_config,
    )



    completions = tokenizer.batch_decode(
        sequence_ids[:, processed_inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]


    # Compute rewards
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    response_lengths = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )
        answer = answer_match.group(1).strip() if answer_match else None

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

# Main training function
def main():
    seed = 42
    wandb_project = "qwen2vl_mathvision_grpo"
    device_index = 0
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    checkpoint_path = Path("/content/drive/MyDrive/tiny-grpo-mllm/checkpoints")
    checkpoint_interval = 50
    train_batch_size = 2
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 4
    rollouts_per_step = 4
    epochs_per_step = 1
    max_norm = 1.0

    max_length = 256
    top_p = 1.0
    temperature = 0.7

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    reference_model, _, _ = load_model(model_name, device_map=device)
    model, tokenizer, processor = load_model(model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    # Load the MathVision dataset
    dataset = load_dataset("MathLLMs/MathVision", split="test")
    prompt_loader = DataLoader(
        dataset,
        batch_size=rollouts_per_step,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)

    with open("/content/drive/MyDrive/tiny-grpo-mllm/thought_processes.txt", "a", encoding="utf-8") as file:
        for k, prompt_batch in enumerate(prompt_loader):
            rollout_returns = []
            rollout_lens = []

            questions = prompt_batch["question"]
            answers = prompt_batch["answer"]
            
            images = prompt_batch["decoded_image"]
            with torch.no_grad():
                for q, a, img in zip(questions, answers, images):
                    image = img
                    sequence_ids, returns, action_mask, completions, response_lengths, ans = rollout(
                        model,
                        tokenizer,
                        processor,
                        image,
                        q,
                        a,
                        num_rollouts=group_size,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    

                    file.write("====================================\n")
                    file.write(f"Question: {q}\n")
                    file.write(f"Oracle Answer: {a}\n")
                    file.write(f"Generated {len(completions)} completions:\n")

                    for i, (completion, reward_val) in enumerate(zip(completions, returns.squeeze().tolist())):
                        # Extract answer again for clarity
                        answer_match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
                        if answer_match:
                            answer = answer_match.group(1).strip()
                        else:
                            answer = completion.strip().splitlines()[-1].strip()

                        #image_filename = f"/content/drive/MyDrive/tiny-grpo-mllm/images/question_{k}_{i}.png"
                        #img.save(image_filename)

                        file.write(f"\n--- Completion #{i+1} ---\n")
                        file.write(f"{completion.strip()}\n")
                        file.write(f"Extracted Answer: {answer}\n")
                        file.write(f"Assigned Reward: {reward_val:.3f}\n")

                        print(f"Question\n{q}\nThink\n{completions[0]}\nAnswer\n{answer}\nOracle Answer\n{a}\n\n")


                    file.write("====================================\n\n")
                    file.flush()
                    os.fsync(file.fileno())

                    #print(f"Question\n{q}\nThink\n{completions[0]}\nAnswer\n{ans}\nOracle Answer\n{a}\n\n")

                    os.sync()

                    rollout_returns.append(returns.cpu())
                    rollout_lens.append(response_lengths.cpu())

                    advantages = group_advantages(returns)
                    attention_mask = sequence_ids != tokenizer.eos_token_id

                    log_probs = sequences_log_probs(
                        model=model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                    )

                    # Implement your GRPO loss and training steps here
                    log_probs_ref = sequences_log_probs(
                        model=reference_model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                    )
                    kl = approx_kl_divergence(
                        log_probs=log_probs,
                        log_probs_ref=log_probs_ref,
                        action_mask=action_mask,
                    )

                    experience = Experience(
                        sequences=sequence_ids,
                        action_log_probs=log_probs,
                        log_probs_ref=log_probs_ref,
                        returns=returns,
                        advantages=advantages,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        kl=kl,
                    )
                    replay_buffer.append(experience.to(cpu_device))

            torch.cuda.empty_cache()
            episode_return_sum = torch.stack(rollout_returns).sum()
            print(f"returns of step {k}: {episode_return_sum:.4f}")
            wandb.log({"returns": episode_return_sum})

            if len(rollout_lens) > 0:
                episode_len_sum = torch.stack(rollout_lens).mean()
            else:
                episode_len_sum = torch.tensor(0.0)  # Prevent error if rollout_lens is empty

            wandb.log({"response_lengths": episode_len_sum})
                   

            experience_sampler = DataLoader(
                replay_buffer,
                batch_size=train_batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=join_experience_batch,
            )

            for step_epoch in range(epochs_per_step):
                model.train()

                for exp in experience_sampler:
                    exp: Experience

                    exp = exp.to(device)

                    optimizer.zero_grad()

                    log_probs = sequences_log_probs(
                        model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                    )

                    loss, kl = objective(log_probs=log_probs, experience=exp)

                    if not loss.isfinite():
                        print(f"Loss not finite, skipping backward, loss={loss}")
                        print(f"experience.advantages={experience.advantages}")
                        continue

                    loss.backward()
                    grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                    wandb.log({"kl": kl, "grad_norm": grad_norm})

                    optimizer.step()

            if (
                checkpoint_path is not None
                and checkpoint_interval is not None
                and (k + 1) % checkpoint_interval == 0
            ):
                model.save_pretrained(checkpoint_path / f"step_{k}")

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")


if __name__ == "__main__":
    main()
