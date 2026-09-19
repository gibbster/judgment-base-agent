# `jev-base-agent` — Calibrated Judgment Primitives for Google ADK

Model-agnostic **System One / Judgment** primitives (`JudgmentAgent`, `JudgmentSwitch`, `JudgmentGuard`, `JudgmentMap`) that make adding **Jev (`model="jev-latest"`)** or any calibrated judgment backend to **Google ADK (`google-adk >= 2.7.0`)** workflows effortless.

Works identically inside:
1. **ADK 2.0 Graph `Workflow` (`from google.adk.workflow import Workflow`)** — emitting `Event(output=..., actions=EventActions(route=..., state_delta=...))` and optional `RequestInput` HITL interrupts.
2. **ADK Composite Agents (`SequentialAgent`, `ParallelAgent`, `LoopAgent`)** — subclassing `BaseAgent` with automatic `state_delta` persistence, `LoopAgent` escalation (`escalate=True`), and sub-agent transfers.
