"""Generate synthetic conversational turn-taking dataset with conflicting persona pause distributions.

SYNTHETIC DATASET GENERATION.
Partitions dev and test splits strictly by USER and by SCENARIO TEMPLATE to prevent leakage.
Produces >= 300 test turns with timed audio events, ground-truth turn ends, and user history sessions.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml

DATA_DIR = Path(__file__).parent / "data"

# Disjoint Scenario Templates: Templates 1-8 for Dev, 9-28 for Test
DEV_TEMPLATES = [
    {
        "id": "tmpl_dev_01",
        "domain": "prescription_refill",
        "complete_utterances": [
            "Can I get a refill on my blood pressure medication please?",
            "I need to authorize a refill for my atorvastatin prescription.",
        ],
        "incomplete_utterances": [
            "I called the pharmacy yesterday about my refill, and...",
            "The prescription bottle says zero refills remaining, so...",
        ],
        "ambiguous_utterances": [
            "I need my lisinopril refill",
            "Checking on the status of my prescription refill",
        ],
    },
    {
        "id": "tmpl_dev_02",
        "domain": "lab_results",
        "complete_utterances": [
            "Has the doctor reviewed my routine blood work from Wednesday?",
            "I am calling to see if my cholesterol test results are ready.",
        ],
        "incomplete_utterances": [
            "I had my blood drawn three days ago, but...",
            "The nurse mentioned the panel might take longer, because...",
        ],
        "ambiguous_utterances": [
            "Looking for my lipid panel results",
            "Inquiring about recent lab work",
        ],
    },
    {
        "id": "tmpl_dev_03",
        "domain": "doctor_message",
        "complete_utterances": [
            "Could you leave an urgent message for Dr. Patel to call me?",
            "Please send a note to my physician regarding my dosage.",
        ],
        "incomplete_utterances": [
            "I experienced some mild dizziness after taking the pill, and...",
            "Dr. Patel wanted me to update her if symptoms changed, so...",
        ],
        "ambiguous_utterances": [
            "Leaving a message for Dr. Patel",
            "Checking if Dr. Patel received my message",
        ],
    },
    {
        "id": "tmpl_dev_04",
        "domain": "clinic_hours",
        "complete_utterances": [
            "What time does the urgent care clinic close this evening?",
            "Are walk-in appointments accepted at the downtown facility today?",
        ],
        "incomplete_utterances": [
            "I was hoping to stop by after my work shift, but...",
            "My daughter started running a fever an hour ago, and...",
        ],
        "ambiguous_utterances": [
            "Checking clinic closing hours",
            "Walk-in clinic schedule inquiry",
        ],
    },
]

TEST_TEMPLATES = [
    {
        "id": "tmpl_test_01",
        "domain": "appointment_booking",
        "complete_utterances": [
            "Can I schedule a follow-up appointment with Dr. Evans for next Tuesday?",
            "I need to book an annual physical exam sometime next month please.",
        ],
        "incomplete_utterances": [
            "I looked at my work calendar for Tuesday morning, but...",
            "Dr. Evans said I should return in six weeks, and...",
        ],
        "ambiguous_utterances": [
            "Scheduling a follow-up appointment with Dr. Evans",
            "Booking an appointment for next week",
        ],
    },
    {
        "id": "tmpl_test_02",
        "domain": "insurance_copay",
        "complete_utterances": [
            "What is my specialist copay under the new Blue Cross plan?",
            "Could you verify whether my deductible has been met for the year?",
        ],
        "incomplete_utterances": [
            "The explanation of benefits showed an unexpected charge, so...",
            "I handed my new insurance card to the receptionist, but...",
        ],
        "ambiguous_utterances": [
            "Checking my insurance copay balance",
            "Inquiring about specialist copay requirements",
        ],
    },
    {
        "id": "tmpl_test_03",
        "domain": "billing_dispute",
        "complete_utterances": [
            "I received an invoice for a test that should have been covered by preventative care.",
            "Can someone in billing explain this out-of-network facility fee?",
        ],
        "incomplete_utterances": [
            "My primary care doctor told me this screening had no out-of-pocket cost, but...",
            "I already paid fifty dollars at the time of service, and...",
        ],
        "ambiguous_utterances": [
            "Disputing an unexpected billing statement fee",
            "Reviewing my recent clinic invoice",
        ],
    },
    {
        "id": "tmpl_test_04",
        "domain": "specialist_referral",
        "complete_utterances": [
            "Has my referral to the cardiology department been approved yet?",
            "I need to check the status of my orthopedics referral from Dr. Miller.",
        ],
        "incomplete_utterances": [
            "My doctor submitted the referral paperwork two weeks ago, but...",
            "The physical therapist told me I need a specialist evaluation, and...",
        ],
        "ambiguous_utterances": [
            "Checking status on specialist referral authorization",
            "Referral to cardiology department",
        ],
    },
    {
        "id": "tmpl_test_05",
        "domain": "medical_records",
        "complete_utterances": [
            "How do I request a digital copy of my vaccination records?",
            "Can you transfer my medical charts to my new provider in Seattle?",
        ],
        "incomplete_utterances": [
            "I signed the HIPAA release form online yesterday, but...",
            "My new employer requires proof of immunization by Friday, and...",
        ],
        "ambiguous_utterances": [
            "Requesting immunization medical records copy",
            "Transferring patient records to another clinic",
        ],
    },
    {
        "id": "tmpl_test_06",
        "domain": "symptom_triage",
        "complete_utterances": [
            "I have had a persistent dry cough and mild fever for four days.",
            "Should I come in to urgent care for a sprained ankle that is swelling?",
        ],
        "incomplete_utterances": [
            "I tried taking ibuprofen and elevating it last night, but...",
            "The shortness of breath only happens when I climb stairs, and...",
        ],
        "ambiguous_utterances": [
            "Reporting persistent cough and fever symptoms",
            "Seeking advice for sprained ankle swelling",
        ],
    },
]


CONTINUATIONS: dict[str, str] = {
    # Dev
    "tmpl_dev_01": "they told me the doctor hasn't submitted the renewal yet.",
    "tmpl_dev_02": "the online portal still hasn't posted the report.",
    "tmpl_dev_03": "I am calling to report a new symptom.",
    "tmpl_dev_04": "I wasn't sure if registration closes early today.",
    # Test
    "tmpl_test_01": "Tuesday afternoon would work much better for my schedule.",
    "tmpl_test_02": "I need to understand what portion of the charge I owe.",
    "tmpl_test_03": "the billing statement listed an unexpected facility fee.",
    "tmpl_test_04": "I haven't received any confirmation call from their office.",
    "tmpl_test_05": "I need to verify the clinic faxed the paperwork.",
    "tmpl_test_06": "the joint is still throbbing and painful this morning.",
}


def load_personas() -> dict[str, Any]:
    with open(DATA_DIR / "personas.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["archetypes"]


def sample_pause_ms(mean: float, std: float, min_ms: float, max_ms: float, jitter: float) -> int:
    val = random.gauss(mean + jitter, std)
    return int(max(min_ms, min(max_ms, val)))


def compute_user_p_continuation(persona_info: dict[str, Any], jitter_ms: float) -> float:
    """Compute individual user continuation probability modulated by user jitter."""
    base_cont_prob = persona_info.get("continuation_prob", 0.40)
    return max(0.05, min(0.95, base_cont_prob + (jitter_ms / 200.0)))


def generate_scenario_turn(
    user_info: dict[str, Any],
    persona_info: dict[str, Any],
    template: dict[str, Any],
    turn_category: str,  # "complete" | "incomplete" | "ambiguous"
    turn_index: int,
    start_t: float = 0.0,
    force_is_turn_complete: bool | None = None,
) -> dict[str, Any]:
    """Generate a single scenario with realistic timed audio events and ground-truth label."""
    user_id = user_info["id"]
    jitter = user_info.get("jitter_ms", 0)
    archetype_key = user_info.get("archetype", "unknown")

    # 1. Select utterance text based on category
    if turn_category == "complete":
        text = random.choice(template["complete_utterances"])
        is_turn_complete = True
        syntactically_complete = True
    elif turn_category == "incomplete":
        text = random.choice(template["incomplete_utterances"])
        is_turn_complete = False
        syntactically_complete = False
    elif turn_category == "ambiguous":
        raw_text = random.choice(template["ambiguous_utterances"])
        text = raw_text if raw_text.endswith((".", "?", "!")) else f"{raw_text}."
        syntactically_complete = True
        if force_is_turn_complete is not None:
            is_turn_complete = force_is_turn_complete
        else:
            user_cont_prob = compute_user_p_continuation(persona_info, jitter)
            is_turn_complete = (random.random() >= user_cont_prob)
    else:
        raise ValueError(f"Unknown turn category: {turn_category}")

    # 2. Timing calculation
    words = text.split()
    speech_duration_ms = len(words) * 350.0

    speech_start_t = start_t
    speech_end_t = speech_start_t + speech_duration_ms

    if is_turn_complete:
        pause_cfg = persona_info["turn_end_pause"]
        silence_ms = sample_pause_ms(
            pause_cfg["mean_ms"], pause_cfg["std_ms"],
            pause_cfg["min_ms"], pause_cfg["max_ms"], jitter,
        )
        ground_truth = "RESPOND"
        turn_end_t = speech_end_t
        resumed_text = None
        full_text = text
        events = [
            {"type": "SpeechStarted", "timestamp_ms": round(speech_start_t, 1)},
            {"type": "SpeechDelta", "text": text, "timestamp_ms": round(speech_start_t + 150.0, 1)},
            {"type": "SpeechEnded", "timestamp_ms": round(speech_end_t, 1)},
        ]
    else:
        pause_cfg = persona_info["mid_thought_pause"]
        silence_ms = sample_pause_ms(
            pause_cfg["mean_ms"], pause_cfg["std_ms"],
            pause_cfg["min_ms"], pause_cfg["max_ms"], jitter,
        )
        ground_truth = "WAIT"
        resumed_text = CONTINUATIONS.get(template["id"], "could you please verify that for me?")
        full_text = f"{text.rstrip('. ')} {resumed_text}"

        resume_start_t = speech_end_t + silence_ms
        resume_dur_ms = len(resumed_text.split()) * 350.0
        resume_end_t = resume_start_t + resume_dur_ms
        turn_end_t = resume_end_t

        events = [
            {"type": "SpeechStarted", "timestamp_ms": round(speech_start_t, 1)},
            {"type": "SpeechDelta", "text": text, "timestamp_ms": round(speech_start_t + 150.0, 1)},
            {"type": "SpeechEnded", "timestamp_ms": round(speech_end_t, 1)},
            {"type": "SpeechStarted", "timestamp_ms": round(resume_start_t, 1)},
            {"type": "SpeechDelta", "text": resumed_text, "timestamp_ms": round(resume_start_t + 150.0, 1)},
            {"type": "SpeechEnded", "timestamp_ms": round(resume_end_t, 1)},
        ]

    return {
        "id": f"turn_{user_id}_{template['id']}_{turn_index:03d}",
        "user_id": user_id,
        "persona_id": persona_info["name"],
        "archetype": archetype_key,
        "scenario_template_id": template["id"],
        "domain": template["domain"],
        "category": turn_category,
        "text": text,
        "full_text": full_text,
        "resumed_text": resumed_text,
        "syntactically_complete": syntactically_complete,
        "is_turn_complete": is_turn_complete,
        "ground_truth": ground_truth,
        "speech_start_t": round(speech_start_t, 1),
        "speech_end_t": round(speech_end_t, 1),
        "silence_ms": silence_ms,
        "turn_end_t": round(turn_end_t, 1),
        "events": events,
        "generation_method": "template",
    }


def generate_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate synthetic dev (15 users, 180 turns), test (30 users, 360 turns), and history datasets."""
    random.seed(42)  # Deterministic dataset generation
    personas = load_personas()

    dev_users: list[dict[str, Any]] = []
    test_users: list[dict[str, Any]] = []

    for archetype_key, p in personas.items():
        for u in p["users"]:
            u_meta = {**u, "archetype": archetype_key, "persona": p}
            if u["split"] == "dev":
                dev_users.append(u_meta)
            else:
                test_users.append(u_meta)

    # 1. Generate History Turns per user (5 history sessions = 10 pauses per user)
    user_histories: list[dict[str, Any]] = []
    for u in dev_users + test_users:
        user_id = u["id"]
        persona_info = u["persona"]
        jitter = u.get("jitter_ms", 0)
        pause_cfg = persona_info["mid_thought_pause"]
        end_cfg = persona_info["turn_end_pause"]

        observed_pauses = []
        for _ in range(5):
            mid_p = sample_pause_ms(pause_cfg["mean_ms"], pause_cfg["std_ms"], pause_cfg["min_ms"], pause_cfg["max_ms"], jitter)
            end_p = sample_pause_ms(end_cfg["mean_ms"], end_cfg["std_ms"], end_cfg["min_ms"], end_cfg["max_ms"], jitter)
            observed_pauses.extend([mid_p, end_p])

        sorted_pauses = sorted(observed_pauses)
        user_histories.append({
            "user_id": user_id,
            "persona_name": persona_info["name"],
            "archetype": u["archetype"],
            "jitter_ms": jitter,
            "observed_pauses_ms": observed_pauses,
            "median_pause_ms": sorted_pauses[len(sorted_pauses) // 2],
            "sample_count": len(observed_pauses),
            "generation_method": "template",
        })

    # 2. Generate Dev Turns (15 dev users x 12 turns = 180 turns)
    dev_turns: list[dict[str, Any]] = []
    turn_idx = 0

    for u in dev_users:
        jitter = u.get("jitter_ms", 0)
        p_user = compute_user_p_continuation(u["persona"], jitter)
        # 6 ambiguous turns per user with stratified quota
        k_cont = int(round(6 * p_user))
        ambig_outcomes = [False] * k_cont + [True] * (6 - k_cont)
        random.shuffle(ambig_outcomes)

        categories = ["ambiguous"] * 6 + ["complete"] * 3 + ["incomplete"] * 3
        random.shuffle(categories)

        # 4 templates x 3 repeats = 12 turns
        tmpl_cycle = (DEV_TEMPLATES * 3)[:12]

        for tmpl, cat in zip(tmpl_cycle, categories):
            turn_idx += 1
            force_c = ambig_outcomes.pop() if cat == "ambiguous" else None
            turn = generate_scenario_turn(u, u["persona"], tmpl, cat, turn_idx, force_is_turn_complete=force_c)
            dev_turns.append(turn)

    # 3. Generate Test Turns (30 test users x 12 turns = 360 turns)
    test_turns: list[dict[str, Any]] = []
    turn_idx = 0

    for u in test_users:
        jitter = u.get("jitter_ms", 0)
        p_user = compute_user_p_continuation(u["persona"], jitter)
        # 6 ambiguous turns per user with stratified quota
        k_cont = int(round(6 * p_user))
        ambig_outcomes = [False] * k_cont + [True] * (6 - k_cont)
        random.shuffle(ambig_outcomes)

        categories = ["ambiguous"] * 6 + ["complete"] * 3 + ["incomplete"] * 3
        random.shuffle(categories)

        # 6 templates x 2 repeats = 12 turns
        tmpl_cycle = (TEST_TEMPLATES * 2)[:12]

        for tmpl, cat in zip(tmpl_cycle, categories):
            turn_idx += 1
            force_c = ambig_outcomes.pop() if cat == "ambiguous" else None
            turn = generate_scenario_turn(u, u["persona"], tmpl, cat, turn_idx, force_is_turn_complete=force_c)
            test_turns.append(turn)

    return dev_turns, test_turns, user_histories


def save_datasets() -> None:
    dev_turns, test_turns, user_histories = generate_dataset()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "dev_turns.jsonl", "w", encoding="utf-8") as f:
        for t in dev_turns:
            f.write(json.dumps(t) + "\n")

    with open(DATA_DIR / "test_turns.jsonl", "w", encoding="utf-8") as f:
        for t in test_turns:
            f.write(json.dumps(t) + "\n")

    with open(DATA_DIR / "user_histories.jsonl", "w", encoding="utf-8") as f:
        for h in user_histories:
            f.write(json.dumps(h) + "\n")

    # Leakage assertion
    dev_users = set(t["user_id"] for t in dev_turns)
    test_users = set(t["user_id"] for t in test_turns)
    dev_templates = set(t["scenario_template_id"] for t in dev_turns)
    test_templates = set(t["scenario_template_id"] for t in test_turns)
    assert len(dev_users & test_users) == 0, f"User leakage detected: {dev_users & test_users}"
    assert len(dev_templates & test_templates) == 0, f"Template leakage detected: {dev_templates & test_templates}"

    print("[generate_dataset] SYNTHETIC Dataset Generated Successfully:")
    print(f"  - Dev turns: {len(dev_turns)} turns ({len(dev_users)} users, 12 turns/user)")
    print(f"  - Test turns: {len(test_turns)} turns ({len(test_users)} users, 12 turns/user)")
    print(f"  - User histories: {len(user_histories)} profiles (10 pause events each)")
    print("  - Leakage check: PASS (0 user overlap, 0 template overlap)")


if __name__ == "__main__":
    save_datasets()

