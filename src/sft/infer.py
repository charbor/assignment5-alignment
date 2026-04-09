from typing import Callable

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams
from datasets import load_dataset


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    sampling_params: SamplingParams,
) -> list[dict[str, float]]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, sampling_params)
    accum: list[dict[str, float]] = []
    for output, truth in zip(outputs, ground_truths, strict=True):
        accum.append(reward_fn(output.outputs[0].text, truth))
    return accum


if __name__ == "__main__":
    # I tested this in a runpod, hence /workspace in the paths

    model = LLM("/workspace/models/Qwen2.5-Math-1.5B")
    prompts: list[str] = []
    ground_truths: list[str] = []

    template = open("cs336_alignment/prompts/r1_zero.prompt").read()
    dataset = load_dataset("openai/gsm8k", "main", revision="e53f048")
    for ex in dataset["test"]:
        prompts.append(template.format(question=ex["question"]))
        ground_truths.append(ex["answer"].split("####")[-1].strip())

    max_prompts = 100  # for debugging
    prompts = prompts[:max_prompts]
    ground_truths = ground_truths[:max_prompts]

    result = evaluate_vllm(
        model,
        r1_zero_reward_fn,
        prompts,
        ground_truths,
        sampling_params=SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=2048,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        ),
    )

    # save results
    import json, os, time

    out_dir = "/root/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/gsm8k_baseline_{int(time.time())}.jsonl"
    with open(out_path, "w") as f:
        for prompt, gt, metrics in zip(prompts, ground_truths, result):
            f.write(
                json.dumps({"prompt": prompt, "ground_truth": gt, "metrics": metrics})
                + "\n"
            )
    print(f"Saved {len(result)} results to {out_path}")
