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
)
from datasets import load_dataset
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
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


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: LlamaForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
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


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def main():
    seed = 42
    wandb_project = "tiny_grpo_gsmk8_v2"
    device_index = 0
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    checkpoint_path = Path("/content/drive/MyDrive/tiny-grpo-gsmk8/output3")
    checkpoint_interval = 50
    train_batch_size = 16
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 8 #12
    rollouts_per_step = 16 #32
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 2048
    top_p = 1.0
    temperature = 1.0

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    reference_model, _ = load_model(model_name, device_map=device)
    model, tokenizer = load_model(model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    """
    # GSM8K DATASET
    dataset = load_dataset("gsm8k", "main")
    train_data = dataset["train"]

    prompt_loader = DataLoader(
        train_data,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )
    """


    prompts = read_prompts(
        "/content/tiny-grpo/data/math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 128
        and x["num_terms"] <= 3
        and x["num_digits"] <= 3,
        max_rows=64 * 1024,
    )
    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )


    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)
    with open("/content/drive/MyDrive/tiny-grpo-gsmk8/thought_processes3.txt", "a", encoding="utf-8") as file:
        for k, prompt_batch in enumerate(prompt_loader):

            file.write(f"Step{k}\n\n")
            rollout_returns = []
            rollout_lens = []


            replay_buffer.clear()

            questions = prompt_batch["question"]
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
                        f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                    )


                    file.write(f"Question\n{q}\nThink\n{completions[0]}\nAnswer\n{ans}\nOracle Answer\n{a}\n\n")

                    os.sync()

                    rollout_returns.append(returns.cpu())
                    rollout_lens.append(response_lengths.cpu())

                    advantages = group_advantages(returns)
                    attention_mask = sequence_ids != pad_token_id

                    log_probs = sequences_log_probs(
                        model=model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                    )
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
