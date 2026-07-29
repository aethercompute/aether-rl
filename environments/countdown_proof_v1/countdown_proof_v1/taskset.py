import ast
import random
from collections import Counter
from collections.abc import Iterator
from fractions import Fraction
from typing import Literal

import verifiers.v1 as vf

SYSTEM_PROMPT = (
    "Solve the countdown arithmetic puzzle. Reply with only one arithmetic expression "
    "using each provided number exactly once. Allowed operators are +, -, *, /, and parentheses."
)


def evaluate_expression(expression: str) -> tuple[Fraction, list[int]]:
    tree = ast.parse(expression.strip(), mode="eval")
    numbers: list[int] = []

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            numbers.append(node.value)
            return Fraction(node.value)
        if not isinstance(node, ast.BinOp) or type(node.op) not in {ast.Add, ast.Sub, ast.Mult, ast.Div}:
            raise ValueError("expression contains unsupported syntax")
        left = evaluate(node.left)
        right = evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ValueError("division by zero")
        return left / right

    return evaluate(tree.body), numbers


class CountdownProofData(vf.TaskData):
    numbers: list[int]
    target: int


class CountdownProofTask(vf.Task[CountdownProofData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    def _score(self, response: str | None) -> tuple[bool, bool, bool]:
        try:
            value, used_numbers = evaluate_expression(response or "")
        except (SyntaxError, ValueError):
            return False, False, False
        numbers_match = Counter(used_numbers) == Counter(self.data.numbers)
        target_match = value == self.data.target
        return True, numbers_match, target_match

    @vf.reward(weight=1.0)
    async def countdown_reward(self, trace: vf.Trace) -> float:
        _, numbers_match, target_match = self._score(trace.last_reply)
        return 0.1 * numbers_match + 0.9 * (numbers_match and target_match)

    @vf.metric
    async def valid_expression(self, trace: vf.Trace) -> float:
        valid, _, _ = self._score(trace.last_reply)
        return float(valid)

    @vf.metric
    async def numbers_used_once(self, trace: vf.Trace) -> float:
        _, numbers_match, _ = self._score(trace.last_reply)
        return float(numbers_match)

    @vf.metric
    async def target_match(self, trace: vf.Trace) -> float:
        _, numbers_match, target_match = self._score(trace.last_reply)
        return float(numbers_match and target_match)


class CountdownProofConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "train"


class CountdownProofTaskset(vf.Taskset[CountdownProofTask, CountdownProofConfig]):
    INFINITE = True

    def load(self) -> Iterator[CountdownProofTask]:
        rng = random.Random(100)
        seen: set[tuple[tuple[int, ...], int]] = set()
        index = 0
        candidate_index = 0
        while True:
            numbers, target = _generate_problem(rng, max_number=10 + candidate_index // 8192)
            key = (tuple(sorted(numbers)), target)
            if key in seen:
                continue
            seen.add(key)
            selected = (candidate_index % 2 == 0) == (self.config.split == "train")
            candidate_index += 1
            if not selected:
                continue
            yield CountdownProofTask(
                CountdownProofData(
                    idx=index,
                    prompt=f"Numbers: {', '.join(map(str, numbers))}\nTarget: {target}",
                    system_prompt=SYSTEM_PROMPT,
                    numbers=numbers,
                    target=target,
                ),
                self.config.task,
            )
            index += 1


def _generate_problem(rng: random.Random, *, max_number: int) -> tuple[list[int], int]:
    while True:
        numbers = [rng.randint(1, max_number) for _ in range(rng.choice((3, 3, 4)))]
        values = [Fraction(number) for number in numbers]
        rng.shuffle(values)
        result = values[0]
        for value in values[1:]:
            operations = [result + value, result - value, result * value]
            if value != 0:
                operations.append(result / value)
            result = rng.choice(operations)
        if result.denominator == 1 and 1 <= result <= 100:
            return numbers, int(result)
