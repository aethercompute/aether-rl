# deepscaler-v1

Rule-graded math tasks using the published DeepScaleR prompt and response format.

The training split comes from
[`agentica-org/DeepScaleR-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset),
pinned to revision `b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29`. The taskset exposes the
problem followed by `Let's think step by step and output the final answer within \boxed{}.`
as one user message. There is no system message and the worked `solution` column is not
exposed to the model.

Rewards are binary. A response must contain an ordered `<think>...</think>` block and a
well-formed boxed final answer after the final `</think>`; `math-verify` then compares the
boxed expression with the gold answer. Six rows with empty answers are excluded and two
malformed gold strings are repaired, leaving 40,309 train tasks. No LLM judge or network
call is used for scoring.

Setting `taskset.dataset = "aime24"` loads the pinned 30-problem AIME 2024 dataset with the
same prompt and scoring protocol for in-run diagnostics.
