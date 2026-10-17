"""B0.5 Latency Diagnosis and Synthetic Dataset Audit.

Time-boxed diagnosis:
1. Free RAM check.
2. Baseline measurement: current dtype, torch.get_num_threads(), input tokens.
3. Thread variation: torch.set_num_threads(6) (6 physical cores on Ryzen 5 5500U).
4. Dtype variation: float32 (if spare RAM >= 3 GB).
5. Dynamic int8 quantization test (if one-liner, else skip).
6. Token breakdown: token counts for state vs prompt template.
7. Dataset audit: silence_ms stats (min, median, max) per label, fixed-800 accuracy by silence bucket, Wilson 95% intervals.
"""

import json
import math
import os
import time
from pathlib import Path

# Ensure cache and offline settings are set
os.environ["HF_HOME"] = "E:\\hf-cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TEMP"] = "E:\\tmp"
os.environ["TMP"] = "E:\\tmp"

import numpy as np
import psutil
import torch
from strands_decider.infer import load_engine
from strands_decider.schema import ChoiceQuestion

MODEL_NAME = "StrandsAgents/strands-decider-2B-hobson-v19"


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # for 95% confidence
    p_hat = successes / total
    denominator = 1 + z**2 / total
    centre_adjusted_probability = p_hat + z**2 / (2 * total)
    adjusted_std_dev = math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)
    lower = (centre_adjusted_probability - z * adjusted_std_dev) / denominator
    upper = (centre_adjusted_probability + z * adjusted_std_dev) / denominator
    return round(max(0.0, lower), 4), round(min(1.0, upper), 4)


def get_free_ram_gb() -> float:
    """Return available system RAM in GB."""
    vm = psutil.virtual_memory()
    return vm.available / (1024**3)


def make_question():
    return ChoiceQuestion(
        instructions="Has the speaker finished their conversational turn, or are they pausing mid-thought and still speaking?",
        criteria={
            "WAIT": "The speaker paused mid-thought, hesitating or trailing off, and intends to continue speaking.",
            "RESPOND": "The speaker finished their complete thought or question and is yielding the turn.",
        },
    )


def measure_median_latency(engine, state_text: str, question: ChoiceQuestion, n_runs: int = 5) -> float:
    """Measure median latency over n_runs warm evaluations."""
    # 1 warmup
    engine.ask(state_text, {"turn": question})
    durations = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        engine.ask(state_text, {"turn": question})
        durations.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(durations))


def run_latency_diagnosis():
    print("=" * 60)
    print("1. LATENCY DIAGNOSIS (AMD Ryzen 5 5500U, CPU)")
    print("=" * 60)

    free_ram = get_free_ram_gb()
    print(f"Initial Available RAM: {free_ram:.2f} GB")

    initial_threads = torch.get_num_threads()
    print(f"Default torch.get_num_threads(): {initial_threads}")

    print("\nLoading engine with default settings...")
    engine = load_engine(MODEL_NAME, device="cpu")

    # Detect current dtype
    current_dtype = getattr(engine.model.config, "torch_dtype", None)
    torso_dtype = getattr(engine.model.torso, "dtype", None)
    if torso_dtype is None and hasattr(engine.model.torso, "parameters"):
        first_param = next(engine.model.torso.parameters(), None)
        if first_param is not None:
            torso_dtype = first_param.dtype
    print(f"Current Config torch_dtype: {current_dtype}")
    print(f"Actual Torso Parameter dtype: {torso_dtype}")

    question = make_question()
    test_state = 'Speaker: "Can you transfer this call over to"\n[Silence: 700 ms]\nPace profile: average pause 700 ms.'

    # Check input token count
    # Let's inspect the tokenizer encoding of state and rendered prompt
    tok = engine.model.tokenizer
    state_tokens = len(tok.encode(test_state))
    print(f"State text word count: {len(test_state.split())} words, {state_tokens} tokens")

    # Run ask and inspect Usage
    res = engine.ask(test_state, {"turn": question})
    input_tokens = res.usage.input_tokens
    print(f"Total input tokens forwarded by engine: {input_tokens} tokens")
    print(f"Prompt template overhead: {input_tokens - state_tokens} tokens")

    # Measurement 1: Baseline (default threads, default dtype)
    print("\n--- Benchmark 1: Baseline (Default threads, default dtype) ---")
    med_baseline = measure_median_latency(engine, test_state, question, n_runs=5)
    print(f"Baseline Median Latency: {med_baseline:.1f} ms")

    # Measurement 2: Set threads to 6 physical cores
    print("\n--- Benchmark 2: Threading (Set to 6 physical cores) ---")
    torch.set_num_threads(6)
    med_6threads = measure_median_latency(engine, test_state, question, n_runs=5)
    print(f"6-threads Median Latency: {med_6threads:.1f} ms")
    torch.set_num_threads(initial_threads)  # restore

    # Measurement 3: Dynamic int8 quantization test
    print("\n--- Benchmark 3: Dynamic INT8 Quantization (One-liner test) ---")
    med_int8 = None
    try:
        # Dynamic quantization on linear layers
        quantized_model = torch.ao.quantization.quantize_dynamic(
            engine.model.torso, {torch.nn.Linear}, dtype=torch.qint8
        )
        # Temporarily swap torso
        orig_torso = engine.model.torso
        engine.model.torso = quantized_model
        med_int8 = measure_median_latency(engine, test_state, question, n_runs=5)
        print(f"Dynamic INT8 Median Latency: {med_int8:.1f} ms")
        engine.model.torso = orig_torso  # restore
    except Exception as e:
        print(f"Dynamic INT8 quantization skipped / failed: {e}")

    # Measurement 4: Float32 dtype
    print("\n--- Benchmark 4: Float32 Dtype ---")
    med_fp32 = None
    free_ram_now = get_free_ram_gb()
    print(f"Available RAM before FP32 load: {free_ram_now:.2f} GB (Requirement: >= 3 GB spare)")
    if free_ram_now < 3.0:
        print("STOPPING FP32 test: insufficient spare RAM (< 3 GB spare).")
    else:
        try:
            # Test converting torso to float32
            print("Converting model torso to float32...")
            engine.model.torso = engine.model.torso.to(torch.float32)
            med_fp32 = measure_median_latency(engine, test_state, question, n_runs=5)
            print(f"FP32 Median Latency: {med_fp32:.1f} ms")
        except Exception as e:
            print(f"FP32 test failed: {e}")

    diagnosis_results = {
        "free_ram_gb": round(free_ram, 2),
        "default_dtype": str(torso_dtype),
        "default_threads": initial_threads,
        "input_tokens_total": input_tokens,
        "state_tokens": state_tokens,
        "prompt_template_overhead_tokens": input_tokens - state_tokens,
        "baseline_median_ms": round(med_baseline, 1),
        "threads_6_median_ms": round(med_6threads, 1),
        "int8_median_ms": round(med_int8, 1) if med_int8 else "SKIPPED",
        "fp32_median_ms": round(med_fp32, 1) if med_fp32 else "SKIPPED",
    }
    return diagnosis_results


def run_dataset_audit():
    print("\n" + "=" * 60)
    print("2. SYNTHETIC DATASET AUDIT (b0_cases.jsonl, n=40)")
    print("=" * 60)

    cases_path = Path("E:/patient-listener/backend/eval/data/b0_cases.jsonl")
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    # Separate by label
    wait_cases = [c for c in cases if c["ground_truth"] == "WAIT"]
    respond_cases = [c for c in cases if c["ground_truth"] == "RESPOND"]

    wait_silences = [c["silence_ms"] for c in wait_cases]
    respond_silences = [c["silence_ms"] for c in respond_cases]

    audit_stats = {
        "WAIT (still_thinking)": {
            "count": len(wait_cases),
            "min_silence_ms": min(wait_silences),
            "median_silence_ms": float(np.median(wait_silences)),
            "max_silence_ms": max(wait_silences),
        },
        "RESPOND (done)": {
            "count": len(respond_cases),
            "min_silence_ms": min(respond_silences),
            "median_silence_ms": float(np.median(respond_silences)),
            "max_silence_ms": max(respond_silences),
        },
    }

    print("Silence distribution by label:")
    for label, stats in audit_stats.items():
        print(f"  {label}: min={stats['min_silence_ms']}ms, median={stats['median_silence_ms']}ms, max={stats['max_silence_ms']}ms (n={stats['count']})")

    # Evaluate fixed 800ms split by silence bucket
    # Buckets: < 800 ms vs >= 800 ms
    bucket_sub800 = [c for c in cases if c["silence_ms"] < 800]
    bucket_gte800 = [c for c in cases if c["silence_ms"] >= 800]

    # For fixed 800:
    # If silence < 800: predicts WAIT
    # If silence >= 800: predicts RESPOND
    def eval_fixed(subset):
        correct = 0
        for c in subset:
            pred = "RESPOND" if c["silence_ms"] >= 800 else "WAIT"
            if pred == c["ground_truth"]:
                correct += 1
        return correct, len(subset), round(correct / len(subset), 4) if subset else 0.0

    corr_sub800, n_sub800, acc_sub800 = eval_fixed(bucket_sub800)
    corr_gte800, n_gte800, acc_gte800 = eval_fixed(bucket_gte800)
    total_corr = corr_sub800 + corr_gte800
    total_n = len(cases)
    total_acc = total_corr / total_n

    print("\nFixed 800ms accuracy by silence bucket:")
    print(f"  Silence < 800 ms  : {corr_sub800}/{n_sub800} ({acc_sub800 * 100:.1f}%)")
    print(f"  Silence >= 800 ms : {corr_gte800}/{n_gte800} ({acc_gte800 * 100:.1f}%)")
    print(f"  Overall Fixed 800 : {total_corr}/{total_n} ({total_acc * 100:.1f}%)")

    # Wilson 95% confidence intervals
    # From b0_results.json: Decider was 34/40 (85.0%), Fixed was 16/40 (40.0%)
    decider_low, decider_high = wilson_score_interval(34, 40)
    fixed_low, fixed_high = wilson_score_interval(16, 40)

    print("\nWilson 95% Confidence Intervals (n=40):")
    print(f"  Strands Decider 2B : 85.0% [95% CI: {decider_low * 100:.1f}% - {decider_high * 100:.1f}%]")
    print(f"  Fixed 800ms        : 40.0% [95% CI: {fixed_low * 100:.1f}% - {fixed_high * 100:.1f}%]")

    # Check for adversarial separation
    is_adversarial = False
    if audit_stats["WAIT (still_thinking)"]["median_silence_ms"] >= 800 and audit_stats["RESPOND (done)"]["median_silence_ms"] < 800:
        is_adversarial = True
        print("\n[CRITICAL AUDIT FINDING]:")
        print("  WAIT cases median silence is >= 800ms (thinking cases were written with long pauses).")
        print("  RESPOND cases median silence is < 800ms (done cases were written with short pauses).")
        print("  This distribution is inherently adversarial to a fixed 800ms threshold.")
        print("  CONCLUSION: The '6x' figure must NOT be cited as a scientific benchmark in the article.")
        print("  Label: SYNTHETIC, n=40, written by the coding agent.")

    audit_results = {
        "label_stats": audit_stats,
        "fixed_800_by_bucket": {
            "silence_lt_800ms": {"correct": corr_sub800, "total": n_sub800, "acc": acc_sub800},
            "silence_gte_800ms": {"correct": corr_gte800, "total": n_gte800, "acc": acc_gte800},
            "overall": {"correct": total_corr, "total": total_n, "acc": total_acc},
        },
        "wilson_95_ci": {
            "decider": {"accuracy": 0.85, "ci_lower": decider_low, "ci_upper": decider_high},
            "fixed_800ms": {"accuracy": 0.40, "ci_lower": fixed_low, "ci_upper": fixed_high},
        },
        "is_adversarial_to_fixed_timeout": is_adversarial,
        "dataset_label": "SYNTHETIC, n=40, written by coding agent",
    }
    return audit_results


def main():
    diag = run_latency_diagnosis()
    audit = run_dataset_audit()

    out_path = Path("E:/patient-listener/backend/eval/results/b0_5_diagnosis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"diagnosis": diag, "audit": audit}, f, indent=2)
    print(f"\nSaved diagnosis & audit results to {out_path}")


if __name__ == "__main__":
    main()
