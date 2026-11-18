import blip_interface
import time
import re

DISTANCE_THRESHOLD = 0.1

def get_edit_info(pair):
    """
    Extract edit information from the document-pair alignment.

    Expected alignment structure:
        {
            "old_status": ["unchanged", "edited", "deleted", ...],
            "new_status": ["unchanged", "edited", "added", ...]
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
        "added_new": added_new
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




def is_positive(task_type, old_answer):
    NOT_FOUND_MARKERS = ["not_found", "i cannot find", "i can't find"]
    a = str(old_answer).strip().lower()
    if any(a.startswith(m) for m in NOT_FOUND_MARKERS):
        return False
    if task_type == "boolean":
        first = re.match(r"[a-z]+", a)
        return bool(first) and first.group(0) == "yes"
    return bool(a) # extraction: any non-refusal answer is “found”

def provenance_answer_aware(provenance, edit, task, old_answer):
    """
    For positive/found answers:
        Use standard provenance overlap.

    For negative/not-found answers:
        Recompute on any document change.

    Assumes:
        - Filter tasks return bool.
        - Extraction tasks return a string.
        - Unanswerable extraction tasks return "NOT_FOUND".
    """

    task_type = task["type"]
    is_found = is_positive(task_type, old_answer)

    if is_found:
        return provenance_overlap(provenance, edit)

    changed = bool(edit["edited_old"] or edit["deleted_old"] or edit["edited_new"] or edit["added_new"])

    reason = (
        "Old answer was not found; document contains changes"
        if changed
        else "Old answer was not found; document has no changes"
    )

    return changed, reason

def run_policy(policy, pair, task, old_answer, provenance, true_new_answer, new_usage):
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
    elif policy == "provenance_aware":
        should_recompute, reason = provenance_answer_aware(provenance, edit, task, old_answer)

    else:
        raise ValueError(f"Unknown policy: {policy}")

    if should_recompute:
        answer, usage = true_new_answer, new_usage
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