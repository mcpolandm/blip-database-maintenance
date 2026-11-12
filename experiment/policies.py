import blip_interface
import time

DISTANCE_THRESHOLD = 0.1

def get_edit_info(pair):
    """
    Extract edit information from the document-pair alignment.

    Expected alignment structure:
        {
            "old_status": ["unchanged", "edited", "deleted", ...],
            "new_status": ["unchanged", "edited", "added", ...],
            "normalized_distance": 0.25,
            "raw_distance": 10
        }
    """

    alignment = pair["alignment"]

    old_status = alignment.get("old_status", [])
    new_status = alignment.get("new_status", [])

    edited_old = {
        i for i, status in enumerate(old_status)
        if status == "edited"
    }

    deleted_old = {
        i for i, status in enumerate(old_status)
        if status == "deleted"
    }

    edited_new = {
        i for i, status in enumerate(new_status)
        if status == "edited"
    }

    added_new = {
        i for i, status in enumerate(new_status)
        if status == "added"
    }

    return {
        "edited_old": edited_old,
        "deleted_old": deleted_old,
        "edited_new": edited_new,
        "added_new": added_new,
        "normalized_distance": alignment.get("normalized_distance"),
        "raw_distance": alignment.get("raw_distance"),
    }

def calculate_normalized_distance(pair):
    alignment = pair["alignment"]
    arcs = alignment["arcs"]

    added = len(alignment.get("added_new", []))
    deleted = alignment["old_status"].count("deleted")

    total_changes = sum(arc["dist_old"] for arc in arcs)
    total_changes += added + deleted

    total_sentences = len(arcs) + added + deleted

    if total_sentences == 0:
        return 0.0

    return total_changes / total_sentences

def recompute_all():
    """Always recompute the answer."""

    return True, "Always recompute"

def text_diff(edit):
    """Recompute whenever any text has changed."""

    changed = (edit["edited_old"] or edit["deleted_old"] or edit["edited_new"] or edit["added_new"])

    return bool(changed), "Document contains changes" if changed else "No text changes"

def distance(pair, threshold=DISTANCE_THRESHOLD):
    """Recompute when normalized edit distance exceeds the threshold."""

    value = calculate_normalized_distance(pair)

    if value is None:
        raise ValueError("Alignment is missing normalized_distance, which is required by the distance policy.")

    recompute = value > threshold

    reason = (
        f"Normalized distance {value} exceeds threshold {threshold}"
        if recompute
        else f"Normalized distance {value} is within threshold {threshold}"
    )

    return recompute, reason

def provenance_overlap(provenance, edit):
    """
    Recompute if an edited or deleted old sentence was part of
    the answer's BLIP provenance.
    """

    changed_old = edit["edited_old"] | edit["deleted_old"]
    overlap = provenance & changed_old

    recompute = bool(overlap)

    reason = (
        f"Changed sentences overlap with provenance: {sorted(overlap)}"
        if recompute
        else "No changed old sentences overlap with provenance"
    )

    return recompute, reason

def run_policy(policy, pair, task, old_answer, provenance, llm):
    """
    Apply one update policy and return its resulting answer and costs.
    """

    start = time.perf_counter()

    edit = get_edit_info(pair)

    if policy == "recompute":
        should_recompute, reason = recompute_all()

    elif policy == "text_diff":
        should_recompute, reason = text_diff(edit)

    elif policy == "distance":
        should_recompute, reason = distance(pair)

    elif policy == "provenance":
        should_recompute, reason = provenance_overlap(provenance, edit)

    else:
        raise ValueError(f"Unknown policy: {policy}")

    if should_recompute:
        answer, usage = blip_interface.get_answer(
            pair["document_new"],
            task["prompt"],
            llm,
        )
    else:
        answer = old_answer
        usage = {
            "llm_calls": 0,
            "usage": {},
        }

    elapsed = time.perf_counter() - start

    return {
        "answer": answer,
        "recomputed": should_recompute,
        "reason": reason,
        "usage": usage,
        "wall_time_seconds": elapsed,
    }