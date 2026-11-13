
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

def validate_sentence_alignment(pair_data):
    pair_id = pair_data.get("pair_id", "unknown")

    if "sentences_old" not in pair_data:
        raise ValueError(f"{pair_id}: missing sentences_old")

    if "alignment" not in pair_data:
        raise ValueError(f"{pair_id}: missing alignment")

    sentences_old = pair_data["sentences_old"]

    if not sentences_old:
        raise ValueError(f"Pair {pair_data.get('pair_id')} has empty sentences_old.")
    
    old_status = pair_data["alignment"].get("old_status")

    if old_status is None:
        raise ValueError(f"{pair_id}: missing alignment.old_status")

    if len(sentences_old) != len(old_status):
        raise ValueError(
            f"{pair_id}: sentences_old has {len(sentences_old)} entries, "
            f"but old_status has {len(old_status)} entries"
        )

    expected_idxs = list(range(len(old_status)))
    actual_idxs = [sentence["blip_idx"] for sentence in sentences_old]

    if actual_idxs != expected_idxs:
        raise ValueError(
            f"{pair_id}: blip_idx does not match old_status positions. "
            f"Expected {expected_idxs}, got {actual_idxs}"
        )

def make_blip_pair(pair_data: dict, task_prompt: str, old_answer: str) -> Pair:
        """Create a BLIP Pair using the sentence indices from the dataset."""

        validate_sentence_alignment(pair_data)

        sentence_data = pair_data["sentences_old"]

        sentence_data = sorted(sentence_data, key=lambda s: int(s["blip_idx"]))

        texts = [s["text"] for s in sentence_data]
        indices = [int(s["blip_idx"]) for s in sentence_data]

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