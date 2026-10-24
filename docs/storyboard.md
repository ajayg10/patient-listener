# Demo video storyboard (target 2:45, hard limit 3:00)

Record at 1080p, browser zoom 110 to 125%, replay speed **1x** for the hero scenario. Close other tabs. Hide terminals, `.env` and anything with keys. Backend and frontend running as in the README, green "real Strands Decider 2B" banner visible.

| Time | On screen | Narration |
|---|---|---|
| 0:00-0:20 | Title card or the demo page header | "Voice agents decide when you're done talking with a silence timeout. Pause to think, and they cut you off. Patient Listener uses Strands Decider 2B to decide instead." |
| 0:20-1:10 | Demo tab, a slow_pauser demo scenario (e.g. `user_slow_test_08`), **Start Replay at 1x** | "Same user, same words, two agents. Left: a fixed 800 ms timeout. Right: Patient Listener." Pause. "She stops after 'last night, but...'. At 800 ms the timeout answers: interrupted. Decider sees an unfinished sentence, p(done) is low, it says WAIT. She finishes, and it answers about half a second later." |
| 1:10-1:30 | Point at the decision feed, the Bedrock reply box and the timeline | "Each row is one checkpoint: silence so far, Decider's probability, its 300 ms decision time on a T4. Only when it says RESPOND does Claude on Amazon Bedrock write the reply. The timeline shows who cut in early." |
| 1:30-1:40 | Click "Log this cut-off as a regret", point at the Speaker profile | "When the timeout cuts someone off, that's logged as a regret and updates their pace profile. Her 'keeps talking after' pause just went up." |
| 1:40-1:55 | Play Demo Set at 4x, then the tally | "This demo set is selected: 10 scenarios where the timeout cuts in. Fixed: 10 cut-ins. Patient Listener: none, answering about 550 ms after the user finishes. Selected cases show the mechanism. The real numbers are next." |
| 1:50-2:25 | Results tab, Point A table, then the Pareto plot | "Across 360 synthetic test turns, the fixed timeout cuts off 18.9%. Decider: 10%. Decider with a per-user pace profile: 8.1% at a 551 ms median. A per-user threshold with no model did best at 4.2%, so personalization matters more than I expected, and I couldn't show Decider beats a tuned heuristic. On my laptop CPU a decision takes 3.2 seconds, so this needs a GPU." |
| 2:25-2:45 | `patient_listener_pattern/README.md` in the repo | "The reusable part is a Strands InterventionHandler: Proceed on RESPOND, Deny on WAIT, decider injected. Drop it into any agent in a few lines." |
| 2:45-2:55 | Live URL and repo URL | "It's live on AWS: the API on Lambda, the site on Amplify, replies from Bedrock. Every number is in the repo." |

## Recording checklist

- [ ] Green real-cache banner showing (not the red MOCK banner)
- [ ] Hero scenario at 1x, other runs at 2x or 4x
- [ ] Say "selected" when showing the demo set
- [ ] Say "synthetic" when showing the numbers
- [ ] No terminal, `.env` or `~/.aws` on screen
- [ ] Under 3:00
