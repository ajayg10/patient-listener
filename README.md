# Patient Listener: Personalized, Regret-Aware Voice Agent Turn-Taking

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Strands Agents](https://img.shields.io/badge/Strands_Agents-v1.0-orange.svg)](https://strandsagents.com)
[![Strands Decider](https://img.shields.io/badge/Strands_Decider-2B-green.svg)](https://github.com/strands-labs/strands-decider)

Patient Listener is a personalized voice agent turn-taking system built for the **AWS Agent Experience Hackathon** using the `#policy` pattern. It couples fast probability estimation from Strands Decider (2B) with personalized cadence thresholds to eliminate premature interruptions during mid-thought pauses without causing excessive delay.

---

## Key Features

- **Personalized Cadence**: Adapts floor-holding thresholds to speaker pace profiles (fast pausers, deliberate/elderly slow pausers, filler-heavy hesitant speakers).
- **Regret-Aware Policy**: Balances premature interruptions ($P(\text{WAIT})$) against conversational delay via asymmetric thresholding.
- **Strands Agents Intervention**: Modular `InterventionHandler` package ([patient_listener_pattern](file:///patient_listener_pattern/)) that plugs into any Strands Agent in under 10 lines.
- **Speculative Virtual Engine**: Asynchronous turn manager with generation counters, cancel-on-speech, and virtual clock evaluation harness.

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/your-org/patient-listener.git
cd patient-listener

# Install dependencies using uv
uv sync
```

### Running the Offline Pattern Example

```bash
uv run python patient_listener_pattern/example.py
```

### Running the Evaluation Suite

```bash
uv run python -m backend.eval.run_eval
```

---

## Evaluation Summary

See [results.md](file:///results.md) for the complete, measured benchmark tables and definitions.
All numbers are generated programmatically from `eval_results.json`.

---

## Reusable Pattern Package

For documentation and usage of the standalone pattern package, see [patient_listener_pattern/README.md](file:///patient_listener_pattern/README.md).
