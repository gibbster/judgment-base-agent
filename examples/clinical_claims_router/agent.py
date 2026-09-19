"""02_clinical_claims_router — Regulated Prior-Authorization & Clinical Claims Router.

Demonstrates `JudgmentSwitch` inside an ADK 2.0 Graph `Workflow` (`from google.adk.workflow import Workflow, START`)
with a strict `confidence_floor=0.75` and `uncertain_route="human_clinical_intake"`.
Any ambiguous or borderline prior-authorization packet is guaranteed to route to a human
clinical intake specialist rather than risking an uncalibrated automated adjudication.
"""

from __future__ import annotations

from typing import Any

from google.adk.apps import App
from google.adk.workflow import START, Workflow

from judgment_base_agent import JudgmentSwitch

clinical_intake_switch = JudgmentSwitch(
    name="clinical_intake_switch",
    instructions="Classify the incoming clinical prior-authorization or insurance claim packet into the appropriate adjudication track",
    routes={
        "fast_track_auto_approve": (
            "Routine, guidelines-concordant outpatient imaging, generic refill, "
            "or preventive procedure with complete ICD-10/CPT documentation"
        ),
        "md_peer_review": (
            "Inpatient surgical admission, off-label specialty biologic/oncology regimen, "
            "or experimental therapy requiring Medical Director peer-to-peer review"
        ),
        "siu_fraud_investigation": (
            "Anomalous provider billing pattern, unbundled surgical CPT modifiers, "
            "duplicate claim submission, or suspected upcoding requiring SIU audit"
        ),
        "human_clinical_intake": (
            "Incomplete clinical notes, conflicting diagnosis codes, or borderline "
            "medical necessity requiring manual RN/specialist intake"
        ),
    },
    confidence_floor=0.75,
    uncertain_route="human_clinical_intake",
    output_key="claims_routing_decision",
)


def fast_track_auto_approve(node_input: Any) -> dict[str, Any]:
    """Automated adjudication track for high-confidence routine claims."""
    return {
        "adjudication_track": "FAST_TRACK_AUTO_APPROVE",
        "sla": "Immediate (< 60 seconds)",
        "disposition": "APPROVED — Prior-authorization number generated automatically.",
        "calibrated_judgment": node_input,
    }


def md_peer_review(node_input: Any) -> dict[str, Any]:
    """Medical Director clinical review queue for complex or specialty regimens."""
    return {
        "adjudication_track": "MD_PEER_TO_PEER_REVIEW",
        "sla": "24-Hour Clinical SLA",
        "disposition": "QUEUED_FOR_ONCOLOGY_OR_SURGICAL_MD — Clinical packet routed to specialty board reviewer.",
        "calibrated_judgment": node_input,
    }


def siu_fraud_investigation(node_input: Any) -> dict[str, Any]:
    """Special Investigations Unit (SIU) hold for anomalous billing or upcoding."""
    return {
        "adjudication_track": "SIU_FRAUD_AND_INTEGRITY_AUDIT",
        "sla": "Pre-Payment Hold",
        "disposition": "HELD_FOR_SIU_AUDIT — Payment suspended pending CPT unbundling / upcoding verification.",
        "calibrated_judgment": node_input,
    }


def human_clinical_intake(node_input: Any) -> dict[str, Any]:
    """Mandatory safe fallback when Jev confidence < 0.75 or documentation is incomplete."""
    return {
        "adjudication_track": "HUMAN_CLINICAL_INTAKE_FALLBACK",
        "sla": "4-Hour RN Triage Queue",
        "disposition": (
            "SAFE_FALLBACK_TRIGGERED — Calibrated confidence below 0.75 floor or documentation "
            "requires registered nurse verification before automated routing."
        ),
        "calibrated_judgment": node_input,
    }


root_agent = Workflow(
    name="clinical_claims_router",
    description=(
        "ADK 2.0 Graph Workflow using JudgmentSwitch (confidence_floor=0.75) to route "
        "clinical prior-authorizations across auto-adjudication, MD review, SIU fraud audit, and RN fallback."
    ),
    edges=[
        (START, clinical_intake_switch),
        (
            clinical_intake_switch,
            {
                "fast_track_auto_approve": fast_track_auto_approve,
                "md_peer_review": md_peer_review,
                "siu_fraud_investigation": siu_fraud_investigation,
                "human_clinical_intake": human_clinical_intake,
            },
        ),
    ],
)

app = App(name="clinical_claims_router", root_agent=root_agent)
