import csv
import json
import re
import statistics

from collections import defaultdict
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RESULTS_DIR = Path("data/results")
OUTPUT_DIR = Path("data/analysis")

NOT_FOUND_MARKERS = (
    "not_found",
    "not found",
    "i cannot find",
    "i can't find",
    "i cannot determine",
    "i can't determine",
    "i don't know",
    "unable to find",
    "unable to determine",
    "not available",
)

BOOLEAN_TASK_TYPES = {"boolean", "filter"}


# --------------------------------------------------
# Answer normalization and comparison
# --------------------------------------------------

def is_not_found(answer):
    if answer is None:
        return True

    a = str(answer).strip().lower()

    if not a:
        return True

    return any(marker in a for marker in NOT_FOUND_MARKERS)


def normalize_boolean(answer):
    """
    Classify Boolean answers as yes, no, or not_found.
    """

    if is_not_found(answer):
        return "not_found"

    a = str(answer).strip().lower()

    match = re.match(r"^(yes|no)\b", a)

    if match:
        return match.group(1)

    # Also support true/false representations.
    match = re.match(r"^(true|false)\b", a)

    if match:
        return "yes" if match.group(1) == "true" else "no"

    return "not_found"


def normalize_extraction(answer):
    """
    Normalize an extraction answer for phrase comparison.
    """

    if is_not_found(answer):
        return "not_found"

    a = str(answer).strip().casefold()

    # Remove punctuation while preserving word boundaries.
    a = re.sub(r"[^\w\s]", " ", a)

    # Normalize whitespace.
    a = " ".join(a.split())

    return a


def normalize_answer(answer, task_type):
    if task_type in BOOLEAN_TASK_TYPES:
        return normalize_boolean(answer)

    if task_type == "extraction":
        return normalize_extraction(answer)

    raise ValueError(f"Unknown task type: {task_type}")


def answers_match(predicted, expected, task_type):
    """
    Compare answers using task-specific rules.

    Extraction uses phrase containment so that a short extracted
    value can match a longer sentence containing that value.
    """

    predicted_normalized = normalize_answer(predicted, task_type)
    expected_normalized = normalize_answer(expected, task_type)

    if predicted_normalized == expected_normalized:
        return True

    # Different not-found classifications should not match.
    if (
        predicted_normalized == "not_found"
        or expected_normalized == "not_found"
    ):
        return False

    if task_type == "extraction":
        # Check whether either answer contains the other as a
        # complete phrase, rather than a partial word.
        shorter, longer = sorted(
            [predicted_normalized, expected_normalized],
            key=len,
        )

        pattern = r"(?<!\w)" + re.escape(shorter) + r"(?!\w)"

        return bool(re.search(pattern, longer))

    return False


# --------------------------------------------------
# Usage and cost extraction
# --------------------------------------------------

def get_usage_totals(usage_data):
    """
    Accept either:
      - A single usage dictionary
      - A list of usage dictionaries

    Return aggregated token counts.
    """

    if isinstance(usage_data, dict):
        usage_data = [usage_data]

    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
    }

    for usage in usage_data or []:
        for key in totals:
            totals[key] += usage.get(key, 0) or 0

    totals["total_tokens"] = (
        totals["prompt_tokens"] + totals["completion_tokens"]
    )

    return totals


def get_call_count(usage_data):
    if not usage_data:
        return 0

    return usage_data.get("llm_calls", 0)


def get_cost(record):
    """
    Extract the different cost components for one task-policy result.

    Shared:
        Initial old-answer generation.

    Policy:
        Answer generation triggered by the policy.

    Provenance:
        Cost of constructing provenance, when applicable.
    """

    shared = record.get("shared_cost", {})

    old_task = shared.get("old_task", {})
    new_task = shared.get("new_task", {})

    old_usage = get_usage_totals(old_task.get("usage", {}))
    new_usage = get_usage_totals(new_task.get("usage", {}))

    old_calls = get_call_count(old_task)
    new_calls = get_call_count(new_task)

    policy = record["policy_result"]

    answer_usage = policy.get("usage", {})
    answer_tokens = get_usage_totals(answer_usage.get("usage", {}))
    answer_calls = get_call_count(answer_usage)

    provenance_usage = policy.get("provenance_usage", {})

    provenance_tokens = get_usage_totals(
        provenance_usage.get("usage", [])
    )

    provenance_calls = get_call_count(provenance_usage)
    provenance_time = provenance_usage.get("wall_time_seconds", 0) or 0

    # Incremental policy cost:
    # Provenance construction + any answer recomputation.
    incremental_calls = provenance_calls + answer_calls

    incremental_tokens = {
        key: provenance_tokens[key] + answer_tokens[key]
        for key in provenance_tokens
    }

    # Lifecycle cost:
    # Initial old answer + incremental policy cost.
    lifecycle_calls = old_calls + incremental_calls

    lifecycle_tokens = {
        key: old_usage[key] + incremental_tokens[key]
        for key in old_usage
    }

    return {
        "old_answer_calls": old_calls,
        "old_answer_prompt_tokens": old_usage["prompt_tokens"],
        "old_answer_completion_tokens": old_usage["completion_tokens"],
        "old_answer_cached_tokens": old_usage["cached_tokens"],
        "old_answer_total_tokens": old_usage["total_tokens"],

        # Oracle cost is reported separately because the new answer
        # was generated in advance for evaluation.
        "oracle_new_answer_calls": new_calls,
        "oracle_new_answer_total_tokens": new_usage["total_tokens"],

        "recompute_calls": answer_calls,
        "recompute_prompt_tokens": answer_tokens["prompt_tokens"],
        "recompute_completion_tokens": answer_tokens["completion_tokens"],
        "recompute_cached_tokens": answer_tokens["cached_tokens"],
        "recompute_total_tokens": answer_tokens["total_tokens"],

        "provenance_calls": provenance_calls,
        "provenance_prompt_tokens": provenance_tokens["prompt_tokens"],
        "provenance_completion_tokens": provenance_tokens["completion_tokens"],
        "provenance_cached_tokens": provenance_tokens["cached_tokens"],
        "provenance_total_tokens": provenance_tokens["total_tokens"],
        "provenance_time_seconds": provenance_time,

        "incremental_llm_calls": incremental_calls,
        "incremental_prompt_tokens": incremental_tokens["prompt_tokens"],
        "incremental_completion_tokens": incremental_tokens["completion_tokens"],
        "incremental_cached_tokens": incremental_tokens["cached_tokens"],
        "incremental_total_tokens": incremental_tokens["total_tokens"],

        "lifecycle_llm_calls": lifecycle_calls,
        "lifecycle_prompt_tokens": lifecycle_tokens["prompt_tokens"],
        "lifecycle_completion_tokens": lifecycle_tokens["completion_tokens"],
        "lifecycle_cached_tokens": lifecycle_tokens["cached_tokens"],
        "lifecycle_total_tokens": lifecycle_tokens["total_tokens"],
    }


# --------------------------------------------------
# Load results
# --------------------------------------------------

def load_results():
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")

    paths = sorted(RESULTS_DIR.glob("*.json"))

    if not paths:
        raise FileNotFoundError(f"No JSON result files found in {RESULTS_DIR}")

    records = []

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            file_results = json.load(f)

        for task_result in file_results.values():
            records.append(task_result)

    print(f"Loaded {len(records)} task results from {len(paths)} files.")

    return records


# --------------------------------------------------
# Per-task analysis
# --------------------------------------------------

def analyze_results(records):
    """
    Create one row per task-policy combination.
    """

    rows = []

    for record in records:
        task_type = record["task_type"]
        true_answer = record["true_new_answer"]

        for policy_name, policy_result in record["policies"].items():

            # Create a temporary record so get_cost can access
            # the shared and policy-specific fields.
            cost_record = {
                "shared_cost": record.get("shared_cost", {}),
                "policy_result": policy_result,
            }

            costs = get_cost(cost_record)

            predicted_answer = policy_result["answer"]

            predicted_normalized = normalize_answer(
                predicted_answer, task_type
            )

            expected_normalized = normalize_answer(
                true_answer, task_type
            )

            row = {
                "pair_id": record["pair_id"],
                "task_id": record["task_id"],
                "task_type": task_type,
                "policy": policy_name,

                "old_answer": record["old_answer"],
                "true_new_answer": true_answer,
                "predicted_answer": predicted_answer,

                "normalized_expected": expected_normalized,
                "normalized_predicted": predicted_normalized,

                "correct": answers_match(
                    predicted_answer,
                    true_answer,
                    task_type,
                ),

                "recomputed": policy_result["recomputed"],
                "reason": policy_result["reason"],

                # This is policy decision time, not answer-generation time.
                "policy_decision_time_seconds": (
                    policy_result.get("wall_time_seconds", 0) or 0
                ),

                **costs,
            }

            rows.append(row)

    return rows


# --------------------------------------------------
# Summary statistics
# --------------------------------------------------

def safe_mean(values):
    return statistics.mean(values) if values else 0


def safe_median(values):
    return statistics.median(values) if values else 0


def summarize(rows):
    """
    Aggregate results by task type and policy.
    """

    groups = defaultdict(list)

    for row in rows:
        groups[(row["task_type"], row["policy"])].append(row)

    summaries = []

    for (task_type, policy), group in sorted(groups.items()):
        n = len(group)

        summary = {
            "task_type": task_type,
            "policy": policy,
            "task_count": n,

            "correct_count": sum(r["correct"] for r in group),
            "accuracy": safe_mean([r["correct"] for r in group]),

            "recomputed_count": sum(r["recomputed"] for r in group),
            "recompute_rate": safe_mean([r["recomputed"] for r in group]),

            "total_incremental_llm_calls": sum(
                r["incremental_llm_calls"] for r in group
            ),
            "mean_incremental_llm_calls": safe_mean(
                [r["incremental_llm_calls"] for r in group]
            ),

            "total_incremental_tokens": sum(
                r["incremental_total_tokens"] for r in group
            ),
            "mean_incremental_tokens": safe_mean(
                [r["incremental_total_tokens"] for r in group]
            ),

            "total_lifecycle_llm_calls": sum(
                r["lifecycle_llm_calls"] for r in group
            ),
            "mean_lifecycle_llm_calls": safe_mean(
                [r["lifecycle_llm_calls"] for r in group]
            ),

            "total_lifecycle_tokens": sum(
                r["lifecycle_total_tokens"] for r in group
            ),
            "mean_lifecycle_tokens": safe_mean(
                [r["lifecycle_total_tokens"] for r in group]
            ),

            "total_provenance_calls": sum(
                r["provenance_calls"] for r in group
            ),
            "mean_provenance_calls": safe_mean(
                [r["provenance_calls"] for r in group]
            ),

            "total_provenance_tokens": sum(
                r["provenance_total_tokens"] for r in group
            ),
            "mean_provenance_tokens": safe_mean(
                [r["provenance_total_tokens"] for r in group]
            ),

            "mean_policy_decision_time_seconds": safe_mean(
                [r["policy_decision_time_seconds"] for r in group]
            ),
            "median_policy_decision_time_seconds": safe_median(
                [r["policy_decision_time_seconds"] for r in group]
            ),

            "mean_provenance_time_seconds": safe_mean(
                [r["provenance_time_seconds"] for r in group]
            ),
            "median_provenance_time_seconds": safe_median(
                [r["provenance_time_seconds"] for r in group]
            ),
        }

        summaries.append(summary)

    return summaries


# --------------------------------------------------
# Boolean confusion matrix
# --------------------------------------------------

def make_boolean_confusion(rows):
    """
    Produce a long-format confusion matrix.

    Each row represents an expected/predicted category pair
    for one policy.
    """

    counts = defaultdict(int)

    for row in rows:
        if row["task_type"] not in BOOLEAN_TASK_TYPES:
            continue

        key = (row["task_type"], row["policy"], row["normalized_expected"], row["normalized_predicted"])

        counts[key] += 1

    output = []

    for key, count in sorted(counts.items()):
        task_type, policy, expected, predicted = key

        output.append({
            "task_type": task_type,
            "policy": policy,
            "expected": expected,
            "predicted": predicted,
            "count": count,
        })

    return output


# --------------------------------------------------
# Save results
# --------------------------------------------------

def save_csv(path, rows):
    if not rows:
        print(f"No rows to save to {path}")
        return

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(rows, summaries, confusion):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_csv(OUTPUT_DIR / "per_task_results.csv", rows)
    save_csv(OUTPUT_DIR / "summary_by_policy.csv", summaries)
    save_csv(OUTPUT_DIR / "boolean_confusion_matrix.csv", confusion)

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print("\nSaved analysis results:")
    print(f"  {OUTPUT_DIR / 'per_task_results.csv'}")
    print(f"  {OUTPUT_DIR / 'summary_by_policy.csv'}")
    print(f"  {OUTPUT_DIR / 'boolean_confusion_matrix.csv'}")
    print(f"  {OUTPUT_DIR / 'summary.json'}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    records = load_results()
    per_task_rows = analyze_results(records)
    summaries = summarize(per_task_rows)
    confusion = make_boolean_confusion(per_task_rows)
    save_outputs(per_task_rows, summaries, confusion)

    print("\nSummary:")
    for summary in summaries:
        print(
            f"{summary['task_type']:12} "
            f"{summary['policy']:20} "
            f"Accuracy: {summary['accuracy']:.3f} "
            f"Recompute: {summary['recompute_rate']:.3f} "
            f"Incremental tokens/task: "
            f"{summary['mean_incremental_tokens']:.1f}"
        )


if __name__ == "__main__":
    main()