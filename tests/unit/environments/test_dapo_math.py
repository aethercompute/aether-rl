import verifiers.v1 as vf
from dapo_math_v1.taskset import extract_final_integer, partition_bucket, thinking_token_count


class PieceTokenizer:
    pieces = {1: "reason", 2: "</", 3: "think>", 4: "Answer", 5: " ignored"}

    def decode(self, token_ids, *, skip_special_tokens=False):
        return "".join(self.pieces[token_id] for token_id in token_ids)

    def encode(self, text, *, add_special_tokens=False):
        assert text == "</think>"
        return [2, 3]


def test_extract_final_integer_requires_exact_boxed_answer_line():
    assert extract_final_integer("Answer: \\boxed{0042}") == 42
    assert extract_final_integer("\nAnswer: \\boxed{-7}\n") == -7
    assert extract_final_integer("Answer: 42") is None
    assert extract_final_integer("Explanation\nAnswer: \\boxed{42}") is None
    assert extract_final_integer("Answer: \\boxed{42} trailing") is None


def test_partition_groups_normalized_prompt_variants():
    assert partition_bucket("Solve  A + B", 16) == partition_bucket("solve a + b", 16)


def test_thinking_tokens_excludes_close_marker_and_final_answer():
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0)),
        nodes=[
            vf.MessageNode(
                message=vf.AssistantMessage(content="Answer", reasoning_content="reason"),
                sampled=True,
                token_ids=[1, 2, 3, 4, 5],
                mask=[True] * 5,
            )
        ],
    )
    assert thinking_token_count(trace, PieceTokenizer()) == 1
