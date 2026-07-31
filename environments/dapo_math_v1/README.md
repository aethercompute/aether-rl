# DAPO Math v1

Single-turn tasks from the pinned `en` configuration of
`open-r1/DAPO-Math-17k-Processed`. The taskset uses the bare `prompt` problem and
adds one user-only instruction requiring `Answer: \boxed{<integer>}` after
reasoning. There is no system prompt.

All published gold solutions are prevalidated as signed integers. Correctness is
strict integer equality after exact final-answer extraction; `math-verify` is not
used because its broader symbolic normalization can over-credit this integer-only
corpus. `thinking_tokens` counts sampled tokens before `</think>` and excludes the
prompt scaffold, closing delimiter, and final answer.

Train and evaluation tasks are partitioned by a stable hash of NFKC-normalized,
case-folded, whitespace-collapsed questions so trivial formatting duplicates cannot
cross the holdout boundary.
