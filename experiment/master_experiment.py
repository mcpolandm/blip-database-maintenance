#for each document:
	#for each task:
		#run task on old
		#run task on new
		#use BLIP on old
		#for each policy:
			#determine if update is needed
			#if yes, rerun
			#record policy and answer (same as old if not rerun)
			#record LLM calls and time for this innermost for loop
		#save info of doc pair id, task type, true new answer, answer for each policy

import json
from pathlib import Path
import time
import blip_interface
import policies
from blip.config import load_config
from blip.llm.client import LLMClient
from blip.rank.embedding import EmbeddingRanker
from blip.pipeline import StrategySpec
from blip.cache.disk import DiskCache

POLICIES = ["recompute", "text_diff", "distance", "provenance", "provenance_aware"]

def load_document_pairs(data_dir: Path) -> list[dict]:
    """Load document-pair JSON files from a directory."""

    paths = sorted(data_dir.glob("*.json"))

    if not paths:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    pairs = []

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            pairs.append(json.load(f))

    return pairs

def run_task(pair, task, strategy, ranker, llm):
    old_answer, old_usage = blip_interface.get_answer(pair["document_old"], task["prompt"], llm)
    print("Got old")
    true_new_answer, new_usage = blip_interface.get_answer(pair["document_new"], task["prompt"], llm)
    print("got new")
    start = time.perf_counter()
    provenance, provenance_usage = blip_interface.get_provenance(pair, task["prompt"], old_answer, llm, strategy, ranker)
    elapsed = time.perf_counter() - start
    print("got prov")
    provenance_usage["wall_time_seconds"] = elapsed

    policy_results = {}
    for policy in POLICIES:
        result = policies.run_policy(policy, pair, task, old_answer, provenance, true_new_answer, new_usage)
        policy_results[policy] = result

    policy_results["provenance"]["provenance_usage"] = provenance_usage
    policy_results["provenance"]["provenance_idxs"] = sorted(provenance)
    policy_results["provenance_aware"]["provenance_usage"] = provenance_usage
    policy_results["provenance_aware"]["provenance_idxs"] = sorted(provenance)

    return {
        "pair_id": pair["pair_id"],
        "task_id": task["task_id"],
        "task": task["prompt"],
        "task_type": task["type"],
        "old_answer": old_answer,
        "true_new_answer": true_new_answer,
        "shared_cost": {
            "old_task": {
                **old_usage,
            },
            "new_task": {
                **new_usage,
            }
        },
        "policies": policy_results
    }


def master():
    pairs = load_document_pairs(Path("data/initial_docs"))

    cfg = load_config()
    llm = LLMClient(cfg)
    ranker = EmbeddingRanker(llm, cache=DiskCache(cfg.cache_dir),)
    strategy = StrategySpec(name="embedding_adaptive_exp", ranker="embedding", scan="adaptive", refine="exp", fastpath="refine",)
    print("Found documents and built BLIP initialization")

    for pair in pairs:
        tasks = pair.get("tasks", [])
        results = {}
        if not tasks:
                raise ValueError(f"Pair {pair['pair_id']} has no tasks.")
        
        for task in tasks:
            task_result = run_task(pair, task, strategy, ranker, llm)
            results[task["task_id"]] = task_result

        output_path = Path("data/results") / f"{pair['pair_id']}.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Saved results for {pair['pair_id']} to {output_path}")

if __name__ == "__main__":
    master()