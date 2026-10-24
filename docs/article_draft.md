# My voice agent kept cutting people off. Here's what a 2B decision model fixed, and what it didn't.

*Draft for the AWS Builder Center. Tags: #strands-decider-2b #policy*

## The problem

Every voice agent makes the same decision dozens of times a minute: the user has gone quiet, so is the turn over? Almost every agent answers with a fixed silence timeout. Pick 600 ms and you cut off anyone who pauses to think ("I looked at my calendar for Tuesday, but..."). Pick 1,200 ms and fast talkers sit in dead air after every sentence.

The words carry the signal a timeout ignores. "but..." means more is coming. "That's all, thanks." means it isn't. And people differ: a pause that means "done" for one speaker is a normal thinking pause for another.

## What I built

Patient Listener is a turn-taking policy for voice agents built on Strands Decider 2B.

- At silence checkpoints (250, 500, 800, 1,200 and 1,800 ms) the turn manager builds a short state: the last ~40 words, the silence so far, and optionally a one-line summary of that user's pause habits from earlier sessions.
- It asks Decider one typed question, `turn_complete`, with two options, WAIT or RESPOND, and gets back a probability.
- A small policy turns that into an action: RESPOND if p(done) is high enough, WAIT otherwise, and a hard max-hold cap of 2,200 ms so it never waits forever.
- Only on RESPOND does the reply agent, a Strands Agent on Amazon Bedrock (Claude Haiku 4.5), start generating.
- Every "you cut me off" is logged as a regret and folded into that user's pace profile, so the next decision for them is more patient.

The decision runs speculatively. A generation counter drops stale answers, and new speech cancels a pending decision, so a slow decision can never make the agent talk over someone who has resumed.

The part meant for reuse is `patient_listener_pattern/`: a Strands `InterventionHandler` that hooks `before_invocation` and `before_model_call`. It returns `Proceed` on RESPOND and `Deny` on WAIT (or `Guide`, if you want a backchannel instead of silence). The decider is injected, so the same handler works with the local model, a remote GPU server or a mock.

## How I evaluated it

I built a synthetic benchmark: three pause-habit archetypes (fast, slow, filler-heavy), several users per archetype with individual jitter, and history sessions per user so a profile has something to learn. Test has 360 turns from 30 users, and dev has 180 turns from 15 users, with no shared users or templates.

I ran the real Decider once over every state the engine could ask about (8,450 states) on a free Colab T4 and replayed all approaches on a virtual clock from that cache. Thresholds were chosen on dev and reported on test, with cluster-bootstrap intervals over users.

## Results

At an equal delay budget (Point A, matched to a fixed 800 ms timeout on dev):

- Fixed 800 ms: 18.9% of turns cut off mid-thought.
- Decider: 10.0%. The paired difference vs fixed 800 is -8.9 points [-15.6, -3.0].
- Decider + pace profile: 8.1% cut-ins at a 551 ms median delay, better than fixed 800 on both axes by point estimate.
- A per-user pause threshold with no model at all: 4.2% at 808 ms.

So the honest summary is that the Decider clearly beats a fixed timeout. Personalization helps a lot, even without a model. Against a tuned text heuristic and the per-user threshold, though, the Decider's paired differences span zero, and I can't claim it adds value over them on this data.

## What went wrong (and what I learned)

**Downloading the model.** The Hugging Face Xet downloader died on Windows ("Background writer channel closed"). The fallback downloader ran out of memory loading the 4.3 GB file. Then curl hit a connection reset at 76%. Resumable `curl -C - --retry-all-errors` finished the job, and I checked the SHA-256 against the Hub.

**CPU latency.** On my laptop CPU (Ryzen 5 5500U) one decision took 3,190 ms. That is longer than my 2,200 ms max-hold cap, so a live CPU run degenerates into a fixed timeout. On a T4 the same call is 235.6 ms median in a benchmark and 301.2 ms over the real evaluation states. If you deploy this, the decision model needs a GPU.

**Two questions are not free.** I hoped to ask `turn_complete` and `mid_thought` in one call. On both CPU and T4 it cost 1.7x to 2.0x a single question, so I kept one live question.

**My first real result was an always-wait policy.** I calibrated thresholds to minimize cut-ins with no delay term. With the real model, every turn waited for the 2,200 ms cap: 0% cut-ins, 2.2 s of dead air. Choosing operating points with an explicit delay budget fixed it.

**Mock results almost fooled me.** Early on, my "Decider" numbers came from a rule-based mock that shared rules with the data generator, and a one-line regex scored 100%. I added provenance to every cache and table so mock numbers can't reach a headline.

**Bedrock access.** On the first AWS account we tried, every Bedrock call failed with `Operation not allowed`, even in the console Playground and after a plan upgrade. On a second account it worked immediately. The demo now streams a real Claude Haiku 4.5 reply from Bedrock as soon as Patient Listener decides to respond, and falls back to a labeled mock reply if a call fails.

**Deploying it.** The API runs on AWS Lambda behind a Function URL, using the Lambda Web Adapter so the replay's streamed events still work, and the frontend is on Amplify Hosting. Two Windows gotchas cost me time: pip evaluates Windows-only dependency markers even when downloading Linux wheels (uv doesn't), and PowerShell's `Compress-Archive` writes backslash paths that Amplify serves as 404s.

## Reuse it

```python
handler = PatientListenerInterventionHandler(decider=my_decider.decide, thresholds=PolicyThresholds(...))
agent = Agent(interventions=[handler])
```

Calibrate the thresholds on your own users. If you only have a CPU, start with the per-user pause threshold. It needs no model and did surprisingly well here.

## Limits

Synthetic text from 60 templates, no real audio, 30 test users, and a dataset with little per-user headroom. Operating points chosen on dev drifted on test. T4 numbers used reference kernels.

Live demo: https://main.d179rqv15h3x1f.amplifyapp.com
Repo: https://github.com/ajayg10/patient-listener
