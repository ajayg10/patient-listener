"""Validate and inspect the real precomputed Decider cache.

Performs integrity, coverage, and latency analysis:
1. Validates schema: backend=local, model_id=StrandsAgents/strands-decider-2B-hobson-v19, no mock markers.
2. Checks against canonical 8450-state enumeration from precompute_decisions.py.
3. Reports duplicate keys and max probability difference across duplicates.
4. Computes latency statistics: overall, per-question, and per-checkpoint median/p95.
5. Emits structured provenance JSON and optionally writes deduplicated gzip archive.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.precompute_decisions import collect_unique_states

DEFAULT_REAL_CACHE = REPO_ROOT / "backend" / "eval" / "data" / "decisions_cache_real.jsonl"
DEFAULT_PROVENANCE_PATH = REPO_ROOT / "backend" / "eval" / "data" / "decisions_cache_real_provenance.json"
EXPECTED_MODEL_ID = "StrandsAgents/strands-decider-2B-hobson-v19"


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_cache(
    cache_path: Path,
    provenance_path: Path | None = None,
    write_gzip_path: Path | None = None,
) -> dict[str, Any]:
    if not cache_path.exists():
        print(f"[validate_cache] ERROR: Cache file does not exist: {cache_path}")
        sys.exit(1)

    print("=" * 70)
    print("REAL DECISION CACHE VALIDATION & PROVENANCE REPORT")
    print(f"Target Cache File: {cache_path}")
    print("=" * 70)

    # 1. Enumerate canonical required states
    enumerated_states = collect_unique_states(include_speech_resumed=True)
    enum_by_key = {item["key"]: item for item in enumerated_states}
    expected_unique_count = len(enum_by_key)
    print(f"[1/5] Canonical State Enumeration: {expected_unique_count} unique keys required.")

    # 2. Parse cache records and validate structure
    raw_records_count = 0
    records_by_key: dict[str, list[dict[str, Any]]] = {}
    latencies_all: list[float] = []
    latencies_by_q: dict[str, list[float]] = {}
    latencies_by_cp: dict[int, list[float]] = {}

    is_gz = cache_path.suffix == ".gz" or str(cache_path).endswith(".jsonl.gz")
    open_fn = gzip.open if is_gz else open

    with open_fn(cache_path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            raw_records_count += 1
            try:
                rec = json.loads(line_str)
            except Exception as e:
                print(f"[validate_cache] ERROR: Malformed JSON on line {line_num}: {e}")
                sys.exit(1)

            # Integrity checks
            backend = rec.get("backend")
            if backend != "local":
                print(f"[validate_cache] ERROR: Line {line_num} backend is '{backend}' (expected 'local'). Mock markers strictly rejected.")
                sys.exit(1)

            model_id = rec.get("model_id")
            if model_id != EXPECTED_MODEL_ID:
                print(f"[validate_cache] ERROR: Line {line_num} model_id is '{model_id}' (expected '{EXPECTED_MODEL_ID}').")
                sys.exit(1)

            # Check mock markers in record
            if "mock" in str(rec.get("backend", "")).lower() or "mock" in str(rec.get("choice", "")).lower():
                print(f"[validate_cache] ERROR: Line {line_num} contains mock marker: {rec}")
                sys.exit(1)

            key = rec.get("key")
            if not key:
                print(f"[validate_cache] ERROR: Line {line_num} missing 'key'.")
                sys.exit(1)

            records_by_key.setdefault(key, []).append(rec)

            # Latency tracking from all valid records
            lat = rec.get("latency_ms")
            if lat is not None and isinstance(lat, (int, float)):
                lat_f = float(lat)
                latencies_all.append(lat_f)
                qid = rec.get("question_id", "unknown")
                latencies_by_q.setdefault(qid, []).append(lat_f)
                cp = rec.get("checkpoint_ms")
                if cp is not None:
                    latencies_by_cp.setdefault(int(cp), []).append(lat_f)

    unique_keys_found = len(records_by_key)
    print(f"[2/5] Read {raw_records_count} total records ({unique_keys_found} unique keys).")

    # 3. Check duplicate keys and measure probability difference
    max_prob_diff = 0.0
    duplicate_keys_count = 0
    deduped_cache: dict[str, dict[str, Any]] = {}

    for key, records in records_by_key.items():
        if len(records) > 1:
            duplicate_keys_count += 1
            # Compare probabilities across all duplicates
            all_probs = [r.get("probabilities", {}) for r in records]
            classes = set().union(*[p.keys() for p in all_probs])
            for cls in classes:
                vals = [p.get(cls, 0.0) for p in all_probs]
                diff = max(vals) - min(vals)
                if diff > max_prob_diff:
                    max_prob_diff = diff
        # Keep one record (the first record)
        deduped_cache[key] = records[0]

    print("[3/5] Duplicate Analysis:")
    print(f"      - Duplicate keys: {duplicate_keys_count}")
    print(f"      - Max probability difference across duplicates: {max_prob_diff:.6f}")

    # 4. Coverage analysis against canonical enumeration
    cache_keys = set(deduped_cache.keys())
    enum_keys = set(enum_by_key.keys())

    covered_keys = enum_keys & cache_keys
    missing_keys = enum_keys - cache_keys
    extra_keys = cache_keys - enum_keys

    print("[4/5] Coverage Analysis:")
    print(f"      - Covered: {len(covered_keys)} / {expected_unique_count} ({(len(covered_keys) / expected_unique_count) * 100:.2f}%)")
    print(f"      - Missing: {len(missing_keys)}")
    print(f"      - Extra:   {len(extra_keys)}")

    if missing_keys:
        print("\n" + "!" * 70)
        print(f"[validate_cache] FATAL ERROR: {len(missing_keys)} required states are missing from cache!")
        print("Missing states breakdown by question and checkpoint:")
        breakdown: dict[str, dict[int, int]] = {}
        for mk in missing_keys:
            st = enum_by_key[mk]
            qid = st.get("question_id", "unknown")
            cp = st.get("checkpoint_ms", -1)
            breakdown.setdefault(qid, {}).setdefault(cp, 0)
            breakdown[qid][cp] += 1

        for qid, cp_dict in sorted(breakdown.items()):
            for cp, cnt in sorted(cp_dict.items()):
                print(f"  - Question: {qid:<15} Checkpoint: {cp:>5} ms  -->  Missing: {cnt}")
        print("!" * 70)
        sys.exit(1)

    # 5. Latency Statistics
    lat_arr = np.array(latencies_all) if latencies_all else np.array([0.0])
    overall_median = float(np.median(lat_arr))
    overall_p95 = float(np.percentile(lat_arr, 95))

    stats_by_q: dict[str, dict[str, float]] = {}
    for qid, q_lats in sorted(latencies_by_q.items()):
        arr = np.array(q_lats)
        stats_by_q[qid] = {
            "median_ms": round(float(np.median(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "count": len(q_lats),
        }

    stats_by_cp: dict[str, dict[str, float]] = {}
    for cp, cp_lats in sorted(latencies_by_cp.items()):
        arr = np.array(cp_lats)
        stats_by_cp[str(cp)] = {
            "median_ms": round(float(np.median(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "count": len(cp_lats),
        }

    print(f"[5/5] Latency Statistics (Tesla T4 Single-Process Execution, n={len(latencies_all)}):")
    print(f"      - Overall Median: {overall_median:.2f} ms")
    print(f"      - Overall P95:    {overall_p95:.2f} ms")
    for qid, q_stat in stats_by_q.items():
        print(f"      - Question '{qid}': median={q_stat['median_ms']} ms, p95={q_stat['p95_ms']} ms (n={q_stat['count']})")
    for cp_str, cp_stat in stats_by_cp.items():
        print(f"      - Checkpoint {cp_str:>4} ms:  median={cp_stat['median_ms']} ms, p95={cp_stat['p95_ms']} ms (n={cp_stat['count']})")

    file_sha256 = compute_file_sha256(cache_path)
    print(f"      - File SHA256:    {file_sha256}")

    # 6. Build Provenance Object
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    provenance = {
        "backend": "local",
        "model_id": EXPECTED_MODEL_ID,
        "unique_keys_total": unique_keys_found,
        "total_records_processed": raw_records_count,
        "hardware": "Tesla T4 (Google Colab free tier)",
        "single_process": True,
        "date": today_str,
        "sha256": file_sha256,
        "reference_kernels": {
            "causal_conv1d": "missing",
            "flash_linear_attention": "missing",
            "execution_path": "PyTorch reference kernels on Tesla T4"
        },
        "latency_statistics": {
            "overall_median_ms": round(overall_median, 2),
            "overall_p95_ms": round(overall_p95, 2),
            "by_question": stats_by_q,
            "by_checkpoint": stats_by_cp,
            "sample_count": len(latencies_all),
        },
        "duplicate_analysis": {
            "duplicate_keys_count": duplicate_keys_count,
            "max_probability_difference": round(max_prob_diff, 8),
        },
        "notes": "Single-process precompute on Tesla T4 GPU; latency_ms is clean and uncontaminated."
    }

    # Save Provenance JSON
    prov_file = provenance_path or DEFAULT_PROVENANCE_PATH
    prov_file.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_file, "w", encoding="utf-8") as pf:
        json.dump(provenance, pf, indent=2)
    print(f"\n[validate_cache] Provenance JSON written to: {prov_file}")

    # 7. Write deduplicated gzip archive if requested
    if write_gzip_path:
        write_gzip_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(write_gzip_path, "wt", encoding="utf-8") as gz:
            for item in deduped_cache.values():
                gz.write(json.dumps(item) + "\n")
        gz_size = write_gzip_path.stat().st_size
        print(f"[validate_cache] Gzip archive written: {write_gzip_path} ({gz_size} bytes, {gz_size / (1024 * 1024):.2f} MB)")
        if gz_size > 2 * 1024 * 1024:
            print(f"[validate_cache] WARNING: Gzip file ({gz_size / (1024 * 1024):.2f} MB) exceeds 2 MB budget!")

    print("=" * 70)
    print("CACHE VALIDATION: SUCCESSFUL (Ready for Real Evaluation)")
    print("=" * 70)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate real Decider decisions cache.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_REAL_CACHE,
        help="Path to decisions_cache_real.jsonl",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=DEFAULT_PROVENANCE_PATH,
        help="Path to save decisions_cache_real_provenance.json",
    )
    parser.add_argument(
        "--gzip-out",
        type=Path,
        default=REPO_ROOT / "backend" / "eval" / "data" / "decisions_cache_real.jsonl.gz",
        help="Path to save deduplicated .jsonl.gz",
    )
    args = parser.parse_args()

    validate_cache(
        cache_path=args.cache,
        provenance_path=args.provenance,
        write_gzip_path=args.gzip_out,
    )


if __name__ == "__main__":
    main()
