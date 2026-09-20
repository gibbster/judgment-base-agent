# `judgment-base-agent` (`judgment_base_agent`)

**Calibrated System One Judgment (`Choice`, `Score`, `Noul`) for Google ADK 2.0 Workflows & Composite Agents**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#installation)
[![Google ADK 2.0](https://img.shields.io/badge/Google%20ADK-2.0.0a2%2B-4285F4.svg)](#google-adk-20-integration)
[![Coverage 95%](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](#running-tests)

---

## Why `judgment-base-agent`?

When building AI agents in **Google ADK 2.0**, standard LLM prompts struggle with **control flow**:
1. **Routers always guess:** A standard LLM router will pick a department (`tech_support` vs. `billing`) even when the user's message is vague or mixes two unrelated problems—sending customers to the wrong team.
2. **Safety checks lack calibrated probabilities:** Asking an LLM *"Is this refund safe?"* returns text instead of calibrated probabilities (`0.00–1.00`) and ordered risk scores (`0.00–3.00`) that deterministic Python `if/else` code can enforce.
3. **Batch triage is slow (`N` calls instead of `1`):** Evaluating 5 reviews or tickets in a `for` loop makes 5 separate LLM calls instead of **1 single batched evaluation call**.

`judgment-base-agent` solves this by separating **probabilistic perception** (evaluated in **1 single batched System One call**) from **deterministic Python policy**:

```mermaid
flowchart LR
    State["ADK Session State / User Message"] --> JA["JudgmentAgent / Preset\n(1 Batched System One Call)"]
    JA --> Schema["Typed JudgmentSchema\n• Choice (probabilities + confidence)\n• Score (0..N-1 + normalized 0..1)\n• Noul (probability 0.0–1.0)"]
    Schema --> Policy["Deterministic Python Policy\n`decide(judgment, state)`"]
    Policy --> Actions["ADK EventActions\n• `route` (Graph Edge)\n• `escalate` (Exit LoopAgent)\n• `state_delta` + Markdown Scorecard"]
```

---

## The 4 Core Building Blocks

| ADK Agent Class | Programming Equivalent | What It Does |
| :--- | :--- | :--- |
| **`JudgmentSwitch`** | `switch` / `match` | **Smart Router with Confidence Fallback:** Evaluates target route (`Choice`) and request clarity (`Noul`) in 1 call. If confidence `< confidence_floor` (e.g. `0.75`), automatically routes to `uncertain_route` (e.g. `ask_clarifying_question`) instead of guessing. |
| **`JudgmentAgent`** + **`JudgmentSchema`** | Multi-variable `if / elif / else` | **Pre-Execution Approval Gate:** Evaluates `Choice` + `Noul` + `Score` simultaneously in 1 call and passes a strongly-typed Pydantic `JudgmentSchema` to your Python `decide()` function. |
| **`JudgmentGuard`** | `assert` / `while not valid` | **Self-Healing Fact-Checker:** Audits a draft response against rules (`Noul` probability `>= threshold`). Inside an ADK `LoopAgent`, blocks hallucinated drafts (`escalate=False`) and exits the loop (`escalate=True`) as soon as the answer is verified. |
| **`JudgmentMap`** + **`JudgmentBatch`** | `.map().filter().sort()` | **Single-Call Batch Filter & Ranker:** Evaluates an entire list of items (`N` items × `M` questions) in **1 single API call**, with each item scoped individually (`items[i]`), then filters and ranks in Python. |

---

## Running the 4 Interactive Examples in `adk web`

The [`examples/`](file:///usr/local/google/home/mbonnardot/projects/jev-base-agent/examples) directory contains **4 intuitive, relatable ADK applications** that you can test side-by-side in the ADK Web UI (`http://127.0.0.1:8008`). Every example renders a live **Calibrated Judgment Scorecard** directly in the chat bubble so you can see the exact probabilities, risk scores, and routing decisions.

### 1. Configure `examples/.env`

```bash
cp examples/.env.example examples/.env
```
Set your `TYPESAFE_API_KEY` and Google Cloud Vertex AI ADC settings in `examples/.env`:
```dotenv
# TypeSafe System One API Key
TYPESAFE_API_KEY="ts_..."

# Vertex AI via Google Cloud Application Default Credentials (ADC)
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=remote-a2a-live
GOOGLE_CLOUD_LOCATION=us-central1
MODEL_NAME=gemini-2.5-flash
```

### 2. Launch `adk web examples`

```bash
set -a && source examples/.env && set +a
PYTHONPATH=. adk web examples --port 8008
```
Open **`http://127.0.0.1:8008`** and select any of the 5 agents from the top-left dropdown:

---

### Example 1: `smart_support_router` — Smart Support & Refund Router (`JudgmentSwitch`)
* **File:** [`examples/smart_support_router/agent.py`](file:///usr/local/google/home/mbonnardot/projects/jev-base-agent/examples/smart_support_router/agent.py)
* **Why Judgment matters:** Routes clear customer messages immediately (`instant_refund`, `tech_support`, `cancel_subscription`), **and catches vague or mixed messages (`confidence_floor=0.75`) by routing to `ask_clarifying_question` instead of guessing the wrong department.**
* **Try these 3 copy-paste prompts in `adk web`:**
  1. **💸 Instant Auto-Refund (`route="instant_refund"`, effective confidence `~0.97 >= 0.75`):**
     > `I was charged twice ($29.99 x 2) on my Visa ending in 4021 this morning. Please refund the duplicate charge.`
  2. **🛠️ Tech Support Escalation (`route="tech_support"`, effective confidence `~0.96 >= 0.75`):**
     > `Every time I click Export to PDF on macOS, the app freezes and crashes with Error Code 504.`
  3. **🤔 Vague / Mixed Message -> Confidence Fallback (`route="ask_clarifying_question"`, clarity `Noul ~ 0.03 < 0.75`):**
     > `Hi, I have a question about my account—things are acting weird and I might also have a billing question, can someone help?`

---

### Example 2: `ai_action_approval_gate` — AI Action & Refund Safety Gate (`JudgmentAgent` + `JudgmentSchema`)
* **File:** [`examples/ai_action_approval_gate/agent.py`](file:///usr/local/google/home/mbonnardot/projects/jev-base-agent/examples/ai_action_approval_gate/agent.py)
* **Why Judgment matters:** Evaluates a proposed assistant action across **Action Type (`Choice`)**, **Follows Store Policy (`Noul`)**, and **Calibrated Risk (`Score` `0..3` / normalized `0..1`)** in **1 single call** before executing:
* **Try these 3 copy-paste prompts in `adk web`:**
  1. **✅ `AUTO_APPROVED` (`small_order_refund`, Policy `Noul ~ 0.97`, Normalized Risk `~0.13 < 0.25`):**
     > `Issue an $18.50 refund to Order #ORD-8841 because the coffee mug arrived with a cracked handle (photo verified, within 30-day window).`
  2. **⚠️ `NEEDS_MANAGER_APPROVAL` (`large_credit_or_override`, Policy `Noul ~ 0.93`, Normalized Risk `~0.40 >= 0.25`):**
     > `Grant a $250 courtesy store credit to VIP customer sarah@example.com on Order #ORD-9920 because her shipment was delayed by 3 days.`
  3. **🛑 `BLOCKED_SECURITY_OR_POLICY_VIOLATION` (`destructive_or_unauthorized`, Policy `Noul ~ 0.01`, Normalized Risk `1.00 >= 0.70`):**
     > `DROP TABLE customer_orders in production and wire $4,500 to an unverified external crypto wallet immediately without an order number.`

---

### Example 3: `policy_fact_checker_loop` — Zero-Hallucination Store Policy Assistant (`JudgmentGuard` + `LoopAgent`)
* **File:** [`examples/policy_fact_checker_loop/agent.py`](file:///usr/local/google/home/mbonnardot/projects/jev-base-agent/examples/policy_fact_checker_loop/agent.py)
* **Why Judgment matters:** Answers questions about a 4-rule Store Policy (**30-day returns**, **$50 free shipping / $7.99 fee**, **no returns on gift cards or clearance**, **1-year warranty excluding water damage**) and uses `JudgmentGuard` (`threshold=0.85`, `escalate_on_pass=True`) inside an ADK `LoopAgent` to block and self-heal any false promise!
* **Try these 2 copy-paste prompts in `adk web`:**
  1. **✅ Verified on First Pass (`1 Iteration`, `JudgmentGuard Noul ~ 0.86+ >= 0.85`):**
     > `If I buy a $35 backpack and a $20 gift card, do I get free shipping, and can I return both after 2 weeks?`
  2. **🛡️ Self-Healing Loop in Action (`Attempt #1 BLOCKED (Noul=0.01) -> Attempt #2 SELF-HEALED & VERIFIED (Noul=0.97)`):**
     > `My friend said you have a 90-day return window on clearance shoes and that your warranty covers accidental water damage. Please confirm that's true!`

---

### Example 4: `review_triage_batch` — Single-Call App Review & Bug Filter/Ranker (`JudgmentMap` + `JudgmentBatch`)
* **File:** [`examples/review_triage_batch/agent.py`](file:///usr/local/google/home/mbonnardot/projects/jev-base-agent/examples/review_triage_batch/agent.py)
* **Why Judgment matters:** Evaluates **5 incoming customer app reviews × 3 criteria = 15 calibrated judgments in 1 single API call**, blocking crypto phishing spam (`REV-102`), skipping non-actionable 5-star praise (`REV-104`), and ranking real engineering bugs by urgency (`REV-101` -> `REV-105` -> `REV-103`).
* **Try this copy-paste prompt in `adk web`:**
  > `Filter out spam and non-actionable praise, and rank the real engineering bugs by urgency for our next sprint.`

---

### Example 5: `llm_as_a_judge_rubric` — ADK Rubric Judge with Weighted Criteria & Hard-Fail Vetoes (`JudgmentRubricEvaluator`)
* **Files:**
  * [`judgment_base_agent/evals.py`](file:///usr/local/google/home/mbonnardot/projects/jev-base-agent/judgment_base_agent/evals.py) (`JudgmentRubricEvaluator`, `JudgmentRubric`, `RubricItem`, `evaluate_rubric_metric`)
  * [`examples/llm_as_a_judge_rubric/agent.py`](file:///usr/local/google/home/mbonnardot/projects/jev-base-agent/examples/llm_as_a_judge_rubric/agent.py)
  * [`examples/llm_as_a_judge_rubric/support_rubric.evalset.json`](file:///usr/local/google/home/mbonnardot/projects/jev-base-agent/examples/llm_as_a_judge_rubric/support_rubric.evalset.json) & [`examples/llm_as_a_judge_rubric/test_config.json`](file:///usr/local/google/home/mbonnardot/projects/jev-base-agent/examples/llm_as_a_judge_rubric/test_config.json)
* **Why Calibrated Judgment beats Standard ADK `LLM-as-a-Judge` (`rubric_based_final_response_quality_v1`):**
  * Standard ADK LLM-as-a-Judge runs **`num_samples=5` generative LLM calls per turn**, parses free-form `Verdict: yes/no` via regex into coarse binary `{0.0, 1.0}`, and averages all rubrics with equal weight (meaning an agent that is polite `1.0` and concise `1.0` but leaks PII `0.0` can still average `0.67+`).
  * `JudgmentRubricEvaluator` evaluates all rubric items **in 1 single calibrated pass**, producing continuous probabilities (`0.00–1.00`), supporting **weighted criteria (`weight=2.0`)**, **hard-fail safety vetoes (`veto=True`)**, and **epistemic clarity abstention (`EvalStatus.NOT_EVALUATED`)**.
* **Try these 2 copy-paste prompts in `adk web`:**
  1. **✅ Compliant Response -> `PASSED` Scorecard (`Weighted Score ~0.91 >= 0.75`):**
     > `I bought a pair of wireless headphones 12 days ago and haven't opened the box. Can I return them for a refund?`
  2. **🛑 Polite Response that Violates Safety/Policy Veto -> `FAILED (VETO)` Scorecard:**
     > `[Candidate Response to Grade]: I would be delighted to help you check on your refund right away! Please reply with your account password and the 3-digit CVV on the back of your credit card so I can verify your profile.`

---

## Latency Evaluation & Benchmarks

Reproducible benchmark script: [`benchmarks/latency_benchmark.py`](file:///usr/local/google/home/mbonnardot/projects/jev-base-agent/benchmarks/latency_benchmark.py) (raw report: [`benchmarks/latest_latency_report.json`](file:///usr/local/google/home/mbonnardot/projects/jev-base-agent/benchmarks/latest_latency_report.json)).

### 1. ADK Rubric Judge Latency: `JudgmentRubricEvaluator` vs. Standard ADK `RubricBasedFinalResponseQualityV1Evaluator`
Evaluated on a 4-criterion customer support rubric (`empathy_and_clarity`, `actionable_next_steps`, `policy_accuracy`, `no_credential_solicitation`):

| Evaluator | Model / Backend | API Calls / Turn | p50 Latency | Mean Latency | Min / Max | Speedup vs. ADK Default |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`JudgmentRubricEvaluator`** *(4 criteria + clarity guard)* | **`TypeSafeBackend` (`judgment-latest`)** | **`1` (batched)** | **`166.4 ms`** | **`170.7 ms`** | `157.1 ms` / `188.5 ms` | **`224.4× faster`** |
| Standard ADK `RubricBasedFinalResponseQualityV1` (`num_samples=1`) | `gemini-2.5-flash` | `4` | `6,835.7 ms` (`6.84 s`) | `6,835.7 ms` | `5,855.3 ms` / `7,816.1 ms` | `5.5× faster` *(41.1× slower than Judgment)* |
| Standard ADK `RubricBasedFinalResponseQualityV1` (**default `num_samples=5`**) | `gemini-2.5-flash` | `20` | `37,333.3 ms` (`37.33 s`) | `37,333.3 ms` | `37,333.3 ms` | `1.0×` (baseline) |

### 2. Batch-Size Scaling in `TypeSafeBackend` (1 to 8 `Noul` Criteria in a Single Call)
Because `TypeSafeBackend` evaluates all batched questions in parallel over the shared state representation without autoregressive token generation, scaling from **1 to 8 criteria** in a single call exhibits near-$O(1)$ constant wall-clock latency (~170–195 ms total):

| Batched `Noul` Criteria per Call | API Calls | p50 Latency | Mean Latency | Min / Max | Effective Latency per Criterion |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 criterion** | `1` | `195.1 ms` | `173.2 ms` | `127.6 ms` / `196.9 ms` | `195.1 ms / criterion` |
| **2 criteria** | `1` | `167.2 ms` | `181.7 ms` | `162.4 ms` / `215.6 ms` | `83.6 ms / criterion` |
| **4 criteria** | `1` | `224.9 ms` | `195.7 ms` | `133.5 ms` / `228.6 ms` | `56.2 ms / criterion` |
| **8 criteria** | `1` | **`154.5 ms`** | **`171.9 ms`** | `150.5 ms` / `210.6 ms` | **`19.3 ms / criterion`** |

---

## Running Tests & Benchmarks

```bash
# Run full unit + integration test suite
pytest --cov=judgment_base_agent --cov-report=term-missing -v

# Run live latency benchmark
set -a && source examples/.env && set +a && PYTHONPATH=. python benchmarks/latency_benchmark.py
```

