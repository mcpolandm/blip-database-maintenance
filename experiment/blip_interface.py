
from dataclasses import asdict, is_dataclass
from blip.pipeline import StrategySpec, run as blip_run
from blip._types import Pair, Sentence
from blip.text.segmenter import segment
from blip.text.tokens import count_tokens
from blip.text.blocks import build_blocks


def usage_to_dict(usage) -> dict:
    """Convert BLIP/Ollama usage information to a serializable dictionary."""

    if usage is None:
        return {}

    if isinstance(usage, dict):
        return usage

    if is_dataclass(usage):
        return asdict(usage)

    if hasattr(usage, "__dict__"):
        return vars(usage)

    return {}

def make_blip_pair(pair_data: dict, task_prompt: str, old_answer: str) -> Pair:
        """Create a BLIP Pair using the sentence indices from the dataset."""

        sentence_data = pair_data.get("sentences_old")

        if sentence_data:
            sentence_data = sorted(
                sentence_data,
                key=lambda s: int(s["blip_idx"]),
            )

            texts = [s["text"] for s in sentence_data]
            indices = [int(s["blip_idx"]) for s in sentence_data]

        else:
            texts = segment(pair_data["document_old"])
            indices = list(range(len(texts)))

        sentences = tuple(
            Sentence(
                idx=idx,
                text=text,
                token_count=count_tokens(text),
            )
            for idx, text in zip(indices, texts)
        )

        blocks = tuple(build_blocks(list(sentences)))

        return Pair(
            pair_id=str(pair_data["pair_id"]),
            doc_id=str(pair_data.get("source", pair_data["pair_id"])),
            question=task_prompt,
            ground_truth=None,
            llm_answer=old_answer,
            sentences=sentences,
            blocks=blocks,
        )

def get_answer(document, task, llm):
    answer, usage = llm.answer(document, task, model="driver")

    return str(answer).strip(), {
        "llm_calls": 1,
        "usage": usage_to_dict(usage),
    }

def get_provenance(pair_data, task, old_answer, llm, strategy, ranker):
    blip_pair = make_blip_pair(pair_data, task, old_answer)
    result = blip_run(blip_pair, strategy, ranker, llm)

    return (
        set(result.provenance_idxs),
        {
            "llm_calls": len(result.usages),
            "usage": [
                usage_to_dict(u)
                for u in result.usages
            ],
        },)