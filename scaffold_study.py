"""Two scaffolds for LAB-Bench ProtocolQA, same model, same dataset, same scorer.

Arm "single": the stock multiple_choice() solver, one call per item.
Arm "chain":  decompose -> answer each sub-question in a fresh context ->
              read the sub-answers -> answer the original question.

See CLAUDE.md for the full experimental design and the rules this file has to
satisfy.
"""

import re

from inspect_ai import Epochs, Task, task
from inspect_ai.model import ChatMessageUser, get_model
from inspect_ai.solver import Generate, Solver, TaskState, multiple_choice, solver
from inspect_evals.lab_bench.lab_bench import (
    MULTIPLE_CHOICE_TEMPLATE,
    DatasetSubsets,
    retrieve_hf_dataset,
)
from inspect_evals.lab_bench.record_to_sample_helpers import (
    UNCERTAIN_ANSWER_CHOICE,
    record_to_sample_protocolqa,
)
from inspect_evals.lab_bench.scorer import precision_choice

SUBQUESTION_TEMPLATE = """
Answer the following question as accurately and concisely as you can, based
on your own knowledge.

{subquestion}
""".strip()

_SUBQUESTION_RE = re.compile(r"(?im)^SUBQUESTION\s+\d+\s*:\s*(.+?)\s*$")

# Same two regexes inspect_ai's own multiple_choice() solver uses to find an
# 'ANSWER: X' line, trimmed down to the single-letter case (lab-bench never
# has multi-correct questions). Reimplemented locally rather than imported
# because inspect_ai.solver._multiple_choice is a private module.
_ANSWER_STRICT_RE = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
_ANSWER_LOOSE_RE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")


def _decompose_template(max_subquestions: int) -> str:
    return f"""
You need to answer the following multiple-choice question, but first break it
into at most {max_subquestions} sub-questions whose answers would help you
solve it.

Question: {{question}}
Options:
{{choices}}

Rules:
- Write between 1 and {max_subquestions} sub-questions, one per line.
- Whoever answers a sub-question will see ONLY that sub-question's text --
  nothing else on this page. They will not see this question, the options
  above, or your other sub-questions. So each sub-question must stand
  entirely on its own: restate any protocol step, number, or detail its
  answer depends on.
- Sub-questions should not depend on each other.
- Do not answer the original question here.

Respond with exactly this format and nothing else:
SUBQUESTION 1: <text>
SUBQUESTION 2: <text>
(up to SUBQUESTION {max_subquestions})
""".strip()


def _parse_subquestions(completion: str, max_subquestions: int) -> list[str]:
    matches = [m.strip() for m in _SUBQUESTION_RE.findall(completion) if m.strip()]
    return matches[:max_subquestions]


def _parse_letter_answer(completion: str, num_choices: int) -> str | None:
    matches = _ANSWER_STRICT_RE.findall(completion) or _ANSWER_LOOSE_RE.findall(completion)
    if not matches:
        return None
    token = matches[-1].strip().rstrip(".").upper().replace(" ", "")
    tokens = [t for t in token.split(",") if t]
    allowed = {chr(ord("A") + i) for i in range(num_choices)}
    if len(tokens) == 1 and tokens[0] in allowed:
        return tokens[0]
    return None


def _classify_answer(state: TaskState) -> tuple[bool, bool]:
    """Returns (parse_failure, declined_insufficient_info) for the model's final answer.

    These are two distinct ways a sample can fail to be a clean answer, and
    they must not be conflated:
      - parse_failure: the completion had no valid 'ANSWER: X' line at all.
      - declined_insufficient_info: it gave a valid letter, and that letter's
        choice text is "Insufficient information to answer the question" --
        a deliberate non-answer, not a formatting failure.
    Only declined_insufficient_info actually scores NOANSWER in
    `precision_choice`: that scorer's no-answer check requires a choice
    marked correct=True whose value equals the no-answer sentinel, so a
    parse_failure sample (no choice marked correct=True at all) falls
    through to its CORRECT/INCORRECT comparison and scores INCORRECT, not
    NOANSWER. `refusal` (the field rule 4 asks for) is still the OR of the
    two -- that's a deliberate, broader definition than the scorer's own
    NOANSWER -- but it only reconciles with the scorer's `coverage` metric
    (which is keyed off NOANSWER) as long as parse_failure stays at zero.
    That held for all 432 samples across the four completed runs, but is
    not guaranteed for any future run: if parse_failure ever fires, `refusal`
    will count it while `coverage` won't, and the two will diverge.
    """
    selected = [i for i, c in enumerate(state.choices) if c.correct is True]
    if not selected:
        return True, False
    declined = any(state.choices[i].value == UNCERTAIN_ANSWER_CHOICE for i in selected)
    return False, declined


def _protocolqa_dataset():
    return retrieve_hf_dataset(DatasetSubsets.ProtocolQA.value, record_to_sample_protocolqa)


@solver
def tag_single_arm(replicate: int) -> Solver:
    """Runs after multiple_choice(); stamps bookkeeping fields shared with the chain arm."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.store.set("arm", "single")
        state.store.set("replicate", replicate)
        state.store.set("decomposition_fallback", False)
        parse_failure, declined = _classify_answer(state)
        state.store.set("parse_failure", parse_failure)
        state.store.set("declined_insufficient_info", declined)
        state.store.set("refusal", parse_failure or declined)
        return state

    return solve


@solver
def decompose_and_answer(max_subquestions: int = 3, replicate: int = 1) -> Solver:
    """Break the question into sub-questions, answer each one fresh, then answer the original."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        original_question = state.user_prompt.text

        # Stage 1: decompose. This call is standalone too (raw model.generate,
        # not state/generate()) but that's incidental here -- nothing has
        # happened yet for there to leak. It sees the question and the choices,
        # so it knows what info is actually needed.
        decompose_prompt = state.choices.prompt(
            original_question, _decompose_template(max_subquestions)
        )
        decompose_output = await model.generate(
            input=[ChatMessageUser(content=decompose_prompt)]
        )
        subquestions = _parse_subquestions(decompose_output.completion, max_subquestions)

        fallback = not subquestions
        state.store.set("decomposition_fallback", fallback)
        state.store.set("subquestions", subquestions)

        if fallback:
            # Same question, same template the single arm uses -- so a
            # fallback item is scored exactly like a single-arm item, not
            # double-penalized for a decomposition that didn't parse.
            final_question = original_question
        else:
            # Stage 2: answer each sub-question in a FRESH context. Each call
            # below builds its own message list containing ONLY that one
            # sub-question's text -- never state.messages, never the
            # decompose call's output, never another sub-question's answer.
            # That message list is thrown away as soon as the call returns;
            # nothing about it is reused for the next sub-question.
            sub_answers: list[str] = []
            for subquestion in subquestions:
                sub_output = await model.generate(
                    input=[
                        ChatMessageUser(
                            content=SUBQUESTION_TEMPLATE.format(subquestion=subquestion)
                        )
                    ]
                )
                sub_answers.append(sub_output.completion.strip())
            state.store.set("sub_answers", sub_answers)

            # Stage 3 setup: assemble what the isolated sub-calls found into
            # a research digest, and fold that into the original question.
            research = "\n\n".join(
                f"Sub-question {i + 1}: {q}\nAnswer: {a}"
                for i, (q, a) in enumerate(zip(subquestions, sub_answers))
            )
            final_question = (
                f"{original_question}\n\n"
                f"Before answering, here is research gathered by investigating "
                f"{len(subquestions)} sub-question(s). Each was investigated by a "
                f"separate, independent instance of you with no knowledge of this "
                f"question, its options, or the other sub-questions:\n\n{research}\n\n"
                f"Use this research where relevant, and your own knowledge "
                f"otherwise, to answer the ORIGINAL question restated above."
            )

        # Stage 3: final answer. This is the one call that goes through the
        # harness `generate()`, using the exact same prompt template and
        # 'ANSWER: $LETTER' instructions as the single arm -- so the only
        # thing that differs between arms is whether a research digest got
        # prepended to the question, not how the final answer is elicited.
        state.user_prompt.text = state.choices.prompt(final_question, MULTIPLE_CHOICE_TEMPLATE)
        state = await generate(state)

        state.store.set("arm", "chain")
        state.store.set("replicate", replicate)

        letter = _parse_letter_answer(state.output.completion, len(state.choices))
        if letter is not None:
            selected = ord(letter) - ord("A")
            for i in range(len(state.choices)):
                state.choices.mark_choice(i, i == selected)

        parse_failure, declined = _classify_answer(state)
        state.store.set("parse_failure", parse_failure)
        state.store.set("declined_insufficient_info", declined)
        state.store.set("refusal", parse_failure or declined)
        return state

    return solve


@task
def single_arm(replicate: int = 1) -> Task:
    """Arm 'single': the stock LAB-Bench ProtocolQA solver, one call per item.

    `replicate` is bookkeeping only (stamped into state.store and into this
    log's task_args) -- pass -T replicate=1 / -T replicate=2 so the two
    independent-sampling runs of this arm are labeled in the log data itself,
    not just distinguished by log filename/timestamp.
    """
    return Task(
        dataset=_protocolqa_dataset(),
        solver=[
            multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE, cot=True),
            tag_single_arm(replicate=replicate),
        ],
        scorer=precision_choice(no_answer=UNCERTAIN_ANSWER_CHOICE),
        epochs=Epochs(1, "mode"),
    )


@task
def chain_arm(max_subquestions: int = 3, replicate: int = 1) -> Task:
    """Arm 'chain': decompose into sub-questions, answer each fresh, then answer the original.

    `replicate` is bookkeeping only, see `single_arm`.
    """
    return Task(
        dataset=_protocolqa_dataset(),
        solver=[decompose_and_answer(max_subquestions=max_subquestions, replicate=replicate)],
        scorer=precision_choice(no_answer=UNCERTAIN_ANSWER_CHOICE),
        epochs=Epochs(1, "mode"),
    )
