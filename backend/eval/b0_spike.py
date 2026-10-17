"""B0 Feasibility Spike Script.

Measures:
1. Cold start model load time on CPU.
2. Warm inference latency (median and p95) across state token sizes (20, 60, 150 tokens).
3. Accuracy and confusion matrix on 40 hand-written pause cases (20 still thinking, 20 done).
4. Comparison against Fixed 800 ms timeout baseline.
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure HF_HOME, TEMP, HF_HUB_DISABLE_XET, and HF_HUB_OFFLINE are set
os.environ["HF_HOME"] = "E:\\hf-cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TEMP"] = "E:\\tmp"
os.environ["TMP"] = "E:\\tmp"


try:
    import numpy as np
    from strands_decider.infer import load_engine
    from strands_decider.schema import ChoiceQuestion
except ImportError as e:
    print(f"Error importing strands_decider: {e}")
    sys.exit(1)


def generate_state_text(target_words: int, prompt_text: str, silence_ms: int) -> str:
    """Generate state text of approximate token/word length."""
    prefix = ""
    if target_words > 20:
        fillers = (
            "Context: Customer calling medical clinic customer support regarding previous claims, "
            "prescription refills, and appointment booking. The dialogue has been ongoing for 3 turns. "
        )
        words = fillers.split()
        prefix = " ".join(words[: max(0, target_words - len(prompt_text.split()) - 6)]) + " "
    return f"{prefix}Speaker: \"{prompt_text}\"\n[Pause silence: {silence_ms} ms]\nPace profile: average pause 650 ms."


def run_latency_benchmark(engine, model_name: str):
    """Benchmark warm inference latency at 20, 60, and 150 tokens."""
    print("\n--- Latency Benchmark (Warm Inference on CPU) ---")
    sizes = [20, 60, 150]
    repeats = 10
    results = {}

    question = ChoiceQuestion(
        instructions="Has the speaker finished their conversational turn, or are they pausing mid-thought and still speaking?",
        criteria={
            "WAIT": "The speaker paused mid-thought, hesitating or trailing off, and intends to continue speaking.",
            "RESPOND": "The speaker finished their complete thought or question and is yielding the turn.",
        },
    )

    for sz in sizes:
        dummy_state = generate_state_text(sz, "Can you transfer this call over to", 700)
        durations = []
        # Warmup for this size
        engine.ask(dummy_state, {"turn": question})

        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = engine.ask(dummy_state, {"turn": question})
            durations.append((time.perf_counter() - t0) * 1000.0)

        median_ms = float(np.median(durations))
        p95_ms = float(np.percentile(durations, 95))
        min_ms = float(np.min(durations))
        max_ms = float(np.max(durations))

        results[f"{sz}_tokens"] = {
            "median_ms": round(median_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "min_ms": round(min_ms, 2),
            "max_ms": round(max_ms, 2),
            "samples": len(durations),
        }
        print(f"State size ~{sz} tokens: median={median_ms:.1f} ms, p95={p95_ms:.1f} ms (min={min_ms:.1f}, max={max_ms:.1f})")

    return results


def evaluate_cases(engine, cases_path: Path):
    """Evaluate 40 cases with Fixed 800ms Timeout vs Strands Decider."""
    print(f"\n--- Evaluating 40 Cases from {cases_path} ---")
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    question = ChoiceQuestion(
        instructions="Has the speaker finished their conversational turn, or are they pausing mid-thought and still speaking?",
        criteria={
            "WAIT": "The speaker paused mid-thought, hesitating or trailing off, and intends to continue speaking.",
            "RESPOND": "The speaker finished their complete thought or question and is yielding the turn.",
        },
    )

    fixed_correct = 0
    fixed_premature = 0  # GT is WAIT, but fixed said RESPOND (silence >= 800)
    fixed_unnecessary_wait = 0  # GT is RESPOND, but fixed said WAIT (silence < 800)

    decider_correct = 0
    decider_premature = 0
    decider_unnecessary_wait = 0
    decider_latencies = []

    case_eval_logs = []

    for c in cases:
        gt = c["ground_truth"]
        silence_ms = c["silence_ms"]
        text = c["text"]

        # 1. Baseline: Fixed 800ms Timeout
        fixed_decision = "RESPOND" if silence_ms >= 800 else "WAIT"
        if fixed_decision == gt:
            fixed_correct += 1
        elif fixed_decision == "RESPOND" and gt == "WAIT":
            fixed_premature += 1
        elif fixed_decision == "WAIT" and gt == "RESPOND":
            fixed_unnecessary_wait += 1

        # 2. Strands Decider
        state_text = f"Speaker: \"{text}\"\n[Silence: {silence_ms} ms]\nPace profile: average pause 700 ms."
        t0 = time.perf_counter()
        ask_res = engine.ask(state_text, {"turn": question})
        lat_ms = (time.perf_counter() - t0) * 1000.0
        decider_latencies.append(lat_ms)

        if hasattr(ask_res, "answers"):
            turn_res = ask_res.answers.get("turn")
        elif isinstance(ask_res, dict):
            turn_res = ask_res.get("answers", {}).get("turn", ask_res.get("turn"))
        else:
            turn_res = ask_res

        chosen = getattr(turn_res, "choice", None)
        probs = getattr(turn_res, "probabilities", None) or {}
        conf = getattr(turn_res, "confidence", None)


        if not chosen:
            # Fallback if result has different attribute
            chosen = str(turn_res)

        decider_decision = chosen.strip().upper()
        if "RESPOND" in decider_decision:
            decider_decision = "RESPOND"
        elif "WAIT" in decider_decision:
            decider_decision = "WAIT"

        if decider_decision == gt:
            decider_correct += 1
        elif decider_decision == "RESPOND" and gt == "WAIT":
            decider_premature += 1
        elif decider_decision == "WAIT" and gt == "RESPOND":
            decider_unnecessary_wait += 1

        case_eval_logs.append({
            "id": c["id"],
            "text": text,
            "silence_ms": silence_ms,
            "gt": gt,
            "fixed": fixed_decision,
            "decider": decider_decision,
            "probs": probs,
            "confidence": conf,
            "latency_ms": round(lat_ms, 1),
        })

    n = len(cases)
    fixed_acc = fixed_correct / n
    decider_acc = decider_correct / n

    results = {
        "num_cases": n,
        "fixed_800ms": {
            "accuracy": round(fixed_acc, 3),
            "correct": fixed_correct,
            "premature_interruptions": fixed_premature,
            "unnecessary_waits": fixed_unnecessary_wait,
        },
        "strands_decider": {
            "accuracy": round(decider_acc, 3),
            "correct": decider_correct,
            "premature_interruptions": decider_premature,
            "unnecessary_waits": decider_unnecessary_wait,
            "latency_median_ms": round(float(np.median(decider_latencies)), 2),
            "latency_p95_ms": round(float(np.percentile(decider_latencies, 95)), 2),
        },
        "bedrock_llm_judge": "SKIPPED — no AWS credentials configured (MOCK_BEDROCK=1 mode per Section 0.5)",
        "cases": case_eval_logs,
    }

    print("\n--- Summary Comparison ---")
    print(f"Fixed 800ms Timeout : Accuracy={fixed_acc * 100:.1f}%, Premature Interruptions={fixed_premature}/{n//2}, Unnecessary Waits={fixed_unnecessary_wait}/{n//2}")
    print(f"Strands Decider 2B   : Accuracy={decider_acc * 100:.1f}%, Premature Interruptions={decider_premature}/{n//2}, Unnecessary Waits={decider_unnecessary_wait}/{n//2}")
    print(f"Decider Latency (CPU): Median={results['strands_decider']['latency_median_ms']} ms, P95={results['strands_decider']['latency_p95_ms']} ms")

    return results


def main():
    model_name = "StrandsAgents/strands-decider-2B-hobson-v19"
    print("=== B0 Feasibility Spike ===")
    print(f"Model: {model_name}")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")

    # (a) Load engine & measure cold start
    print("\n(a) Loading model engine on CPU...")
    t_start = time.perf_counter()
    engine = load_engine(model_name, device="cpu")
    cold_start_s = round(time.perf_counter() - t_start, 2)
    print(f"Cold start model load time: {cold_start_s} seconds")

    # (b) Latency benchmark
    bench_results = run_latency_benchmark(engine, model_name)

    # (c) Evaluate 40 cases
    cases_file = Path(__file__).resolve().parent / "data" / "b0_cases.jsonl"
    eval_results = evaluate_cases(engine, cases_file)

    # Save results
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "b0_results.json"
    full_output = {
        "model_name": model_name,
        "device": "cpu",
        "cold_start_seconds": cold_start_s,
        "benchmarks": bench_results,
        "evaluation": eval_results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nSaved detailed B0 results to {out_file}")


if __name__ == "__main__":
    main()
