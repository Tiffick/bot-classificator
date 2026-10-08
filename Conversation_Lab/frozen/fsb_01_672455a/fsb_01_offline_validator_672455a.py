"""Build and offline-validate the refrozen FSB-01 selected-only package.

The builder uses production Discovery contracts and the deterministic StateApplier,
inspects the CognitiveCore wire schema, but never calls CognitiveCore or performs
network I/O.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.cognitive_turn import (  # noqa: E402
    ActionContent,
    ActionSubjectReference,
    ActionTarget,
    CognitiveTurnResult,
    DecisionIntent,
    EvidenceOrigin,
    ItemOperation,
    ItemOperationKind,
    ReconciliationOutcome,
    ReplySegment,
    SourceSpan,
    StatePatch,
    SystemAction,
    TargetResolution,
)
import ai.discovery_data_model as discovery_model_module  # noqa: E402
from ai.discovery_data_model import (  # noqa: E402
    ActiveContentKind,
    ActiveConversationState,
    DialogueHistory,
    DialogueMessage,
    DialogueRole,
    HumanModel,
    ModelItemKind,
    TargetInteractionKind,
    TargetSubjectKind,
    validate_discovery_memory,
)
from ai.discovery_state_applier import apply_cognitive_turn  # noqa: E402
from ai.cognitive_core import COGNITIVE_TURN_JSON_SCHEMA  # noqa: E402


HEAD = "672455a"
PACKAGE_DIR = Path(__file__).resolve().parent
OLD_PACKAGE_DIR = PACKAGE_DIR.parent / "fsb_01"
CASES_PATH = PACKAGE_DIR / f"fsb_01_cases_{HEAD}.json"
CHALLENGERS_PATH = PACKAGE_DIR / f"fsb_01_challengers_{HEAD}.json"
MANIFEST_PATH = PACKAGE_DIR / f"fsb_01_manifest_{HEAD}.json"
PLAN_PATH = PACKAGE_DIR / f"fsb_01_measurement_plan_{HEAD}.md"
PROMPT_PATH = PACKAGE_DIR / f"fsb_01_evaluator_prompt_{HEAD}.txt"
SCHEMA_PATH = PACKAGE_DIR / f"fsb_01_evaluator_schema_{HEAD}.json"
REPORT_PATH = PACKAGE_DIR / f"fsb_01_freeze_report_{HEAD}.md"
FIXED_TIME = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
CASE_ORDER = ("FSB-H", "FSB-M", "FSB-R", "FSB-S", "FSB-T", "FSB-A")
LEGACY_STATE_HASHES = {
    "E9F54D5806D61AF3952388A53DE4807E21BA970182B4285E4515000CC12C6CD6",
    "C93D283943D135AD3CF740181BD684A42640D2B757A375663CF206E79AAC4225",
    "90E2ECE68C97E793A27449CA9F4195EE0BB1CF8682EBF8C996F0B3C7196B021D",
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_value(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest().upper()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def stable_id(case_id: str, prefix: str, label: str) -> str:
    return f"{prefix}_{uuid5(NAMESPACE_URL, f'fsb-01/{case_id}/{prefix}/{label}').hex}"


class IdFactory:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.counts: dict[str, int] = {}

    def __call__(self, prefix: str) -> str:
        number = self.counts.get(prefix, 0) + 1
        self.counts[prefix] = number
        return stable_id(self.case_id, prefix, str(number))

    def uuid4(self):
        number = self.counts.get("uuid", 0) + 1
        self.counts["uuid"] = number
        return uuid5(NAMESPACE_URL, f"fsb-01/{self.case_id}/uuid/{number}")


@contextmanager
def deterministic_lifecycle(case_id: str):
    ids = IdFactory(case_id)
    # Every production make_id() implementation, including Pydantic default
    # factories captured at class definition time, resolves uuid4 from this
    # module at call time.  Patching that entropy source makes an otherwise
    # ordinary StateApplier replay byte-reproducible without changing its code.
    with patch.object(discovery_model_module, "uuid4", ids.uuid4), patch.object(
        discovery_model_module, "utc_now", lambda: FIXED_TIME
    ):
        yield


def message(case_id: str, role: DialogueRole, text: str, sequence: int) -> DialogueMessage:
    return DialogueMessage(
        id=stable_id(case_id, "msg", str(sequence)),
        role=role,
        text=text,
        sequence=sequence,
        created_at=FIXED_TIME,
    )


def action(
    case_id: str,
    turn: int,
    contents: list[dict[str, str]],
) -> SystemAction:
    action_contents = []
    targets = []
    for index, spec in enumerate(contents, 1):
        local_content = f"content:t{turn}_{index}"
        local_target = f"target:t{turn}_{index}"
        action_contents.append(
            ActionContent(
                local_id=local_content,
                kind=ActiveContentKind(spec["kind"]),
                semantic_content=spec["semantic_content"],
            )
        )
        targets.append(
            ActionTarget(
                local_id=local_target,
                active_content_local_id=local_content,
                subject=ActionSubjectReference(
                    kind=TargetSubjectKind.ACTIVE_CONTENT,
                    active_content_local_id=local_content,
                ),
                interaction=TargetInteractionKind(spec["interaction"]),
            )
        )
    return SystemAction(contents=tuple(action_contents), response_targets=tuple(targets))


def make_turn_result(
    *,
    case_id: str,
    turn: int,
    user_text: str,
    reply: str,
    previous_acs: ActiveConversationState | None,
    decision_intent: str,
    additions: list[dict[str, str]],
    contents: list[dict[str, str]],
    reconciliation_outcome: str | None = None,
) -> CognitiveTurnResult:
    span = SourceSpan(char_start=0, char_end=len(user_text))
    operations = tuple(
        ItemOperation(
            operation=ItemOperationKind.ADD,
            local_id=f"item:t{turn}_{index}",
            kind=ModelItemKind(spec["kind"]),
            content=spec["content"],
            evidence_origin=EvidenceOrigin.CURRENT_USER_MATERIAL,
            source_span=span,
        )
        for index, spec in enumerate(additions, 1)
    )
    resolutions: tuple[TargetResolution, ...] = ()
    if previous_acs is not None:
        previous_ids = tuple(previous_acs.response_targets)
        if len(previous_ids) != 1:
            raise AssertionError("Replay steps may only follow one previous target.")
        previous_target = next(iter(previous_acs.response_targets.values()))
        outcome = reconciliation_outcome
        if outcome is None:
            outcome = (
                ReconciliationOutcome.SUPPORTED.value
                if previous_target.interaction == TargetInteractionKind.EVALUATION
                else ReconciliationOutcome.ANSWERED.value
            )
        resolutions = (
            TargetResolution(
                previous_response_target_ids=previous_ids,
                outcome=ReconciliationOutcome(outcome),
                source_span=span,
            ),
        )
    system_action = action(case_id, turn, contents)
    return CognitiveTurnResult(
        decision_intent=DecisionIntent(decision_intent),
        state_patch=StatePatch(item_operations=operations),
        reconciliation=resolutions,
        system_action=system_action,
        reply_segments=(
            ReplySegment(
                local_id=f"segment:t{turn}",
                text=reply,
                realizes_action_content_ids=tuple(
                    content.local_id for content in system_action.contents
                ),
            ),
        ),
    )


RECIPES: dict[str, dict[str, Any]] = {
    "FSB-H": {
        "intent": "human_discovery",
        "purpose": "Mechanism is already useful; a new Before→Now change makes Personal Meaning decision-relevant.",
        "steps": [
            {
                "user": "После перехода на сменную работу я набрал вес.",
                "reply": "Что обычно происходит с едой после поздней смены?",
                "decision": "mechanism_discovery",
                "adds": [
                    {"kind": "life_change", "content": "переход на сменную работу"},
                    {"kind": "now", "content": "после смены работы набрал вес"},
                ],
                "contents": [{"kind": "system_question", "interaction": "open_response", "semantic_content": "Уточнить повторяющийся пищевой контекст после поздней смены."}],
            },
            {
                "user": "После поздней смены прихожу очень голодный и почти сразу съедаю большой ужин.",
                "reply": "Это основной повторяющийся момент, который мешает есть меньше?",
                "decision": "mechanism_discovery",
                "adds": [{"kind": "barrier", "content": "сильный голод после поздней смены приводит к большому ужину"}],
                "contents": [{"kind": "system_question", "interaction": "open_response", "semantic_content": "Проверить значимость установленного механизма для предыдущих попыток."}],
            },
            {
                "user": "Да, именно поздний голод после смены чаще всего срывает мои попытки есть меньше.",
                "reply": "А что изменилось для вас в жизни по сравнению с тем, как было раньше?",
                "decision": "human_discovery",
                "adds": [{"kind": "previous_attempt", "content": "попытки есть меньше срываются из-за позднего голода"}],
                "contents": [{"kind": "system_question", "interaction": "open_response", "semantic_content": "Открыть значимое сравнение Before→Now после понятого механизма."}],
            },
        ],
        "current": "Раньше я после работы спокойно встречался с друзьями, а теперь избегаю встреч: мне неловко из-за того, как сидит одежда.",
    },
    "FSB-M": {
        "intent": "mechanism_discovery",
        "purpose": "Personal Meaning is established; two barrier explanations remain discriminable from available experience.",
        "steps": [
            {
                "user": "Мне важно снова чувствовать, что я управляю своим вечером, а не еда мной.",
                "reply": "Что вы уже пробовали, чтобы вернуть это ощущение управления?",
                "decision": "mechanism_discovery",
                "adds": [{"kind": "personal_meaning", "content": "важно вернуть ощущение управления своим вечером"}],
                "contents": [{"kind": "system_question", "interaction": "open_response", "semantic_content": "Уточнить значимые предыдущие попытки."}],
            },
            {
                "user": "Пробовал заранее готовить ужин, но в последние два раза всё равно переел.",
                "reply": "Что различалось в обстоятельствах этих двух случаев?",
                "decision": "mechanism_discovery",
                "adds": [{"kind": "previous_attempt", "content": "заранее готовил ужин, но дважды всё равно переел"}],
                "contents": [{"kind": "system_question", "interaction": "open_response", "semantic_content": "Найти наблюдаемое различие между возможными механизмами."}],
            },
        ],
        "current": "В одном случае я пропустил обед, а в другом нормально поел днём, но перед ужином был тяжёлый разговор.",
    },
    "FSB-R": {
        "intent": "reflection",
        "purpose": "Meaning, change, attempt and barrier form a connected picture; another local question has low marginal value.",
        "steps": [
            {
                "user": "После смены работы привычный режим развалился, и я стал набирать вес.",
                "reply": "Что в этом изменении задевает вас сильнее всего?",
                "decision": "human_discovery",
                "adds": [
                    {"kind": "life_change", "content": "после смены работы привычный режим развалился"},
                    {"kind": "now", "content": "после изменения режима стал набирать вес"},
                ],
                "contents": [{"kind": "system_question", "interaction": "open_response", "semantic_content": "Понять личное значение изменения режима и веса."}],
            },
            {
                "user": "Мне неловко встречаться с людьми: одежда сидит иначе, и я всё чаще отказываюсь от встреч.",
                "reply": "Что вы уже пробовали изменить самостоятельно?",
                "decision": "mechanism_discovery",
                "adds": [{"kind": "personal_meaning", "content": "неловкость из-за одежды приводит к отказу от встреч"}],
                "contents": [{"kind": "system_question", "interaction": "open_response", "semantic_content": "Уточнить предыдущую попытку и существенный барьер."}],
            },
        ],
        "current": "Пробовал просто есть меньше, но после поздней смены сильный голод быстро возвращает меня к прежнему режиму.",
    },
    "FSB-S": {
        "intent": "stop_exploration",
        "purpose": "The user twice establishes that further observation is unavailable; the semantic axis is exhausted.",
        "steps": [
            {
                "user": "По вечерам иногда ем больше, чем собирался.",
                "reply": "Что обычно запускает этот момент вечером?",
                "decision": "mechanism_discovery",
                "adds": [{"kind": "experience", "content": "по вечерам иногда ест больше, чем собирался"}],
                "contents": [{"kind": "system_question", "interaction": "open_response", "semantic_content": "Уточнить доступное наблюдение о начале вечернего эпизода."}],
            },
            {
                "user": "Не знаю, я не замечал, с чего это начинается.",
                "reply": "Может быть, вы замечали, что происходит непосредственно перед желанием поесть?",
                "decision": "mechanism_discovery",
                "adds": [{"kind": "other_relevant_user_material", "content": "не наблюдал начало вечернего эпизода"}],
                "contents": [{"kind": "system_question", "interaction": "open_response", "semantic_content": "Проверить наличие другого доступного наблюдения на той же оси."}],
            },
        ],
        "current": "Тоже не знаю. Я этого не отслеживал, и больше деталей у меня нет.",
    },
    "FSB-T": {
        "intent": "transition",
        "purpose": "A coherent sufficient picture has been reflected; the user confirms it and asks how to move forward.",
        "steps": [
            {
                "user": "Я хочу снова спокойно носить привычную одежду и не избегать фотографий.",
                "reply": "Что вы уже пробовали и что обычно мешало продолжать?",
                "decision": "mechanism_discovery",
                "adds": [
                    {"kind": "desired_change", "content": "снова спокойно носить привычную одежду"},
                    {"kind": "personal_meaning", "content": "не хочет избегать фотографий"},
                ],
                "contents": [{"kind": "system_question", "interaction": "open_response", "semantic_content": "Понять значимую попытку и барьер перед завершением Discovery."}],
            },
            {
                "user": "Пытался есть меньше, но после поздних смен сильный голод возвращает меня к большим ужинам.",
                "reply": "Правильно ли я понимаю общую картину: вы хотите вернуть свободу в одежде и фотографиях, а попытки есть меньше срывает сильный голод после поздних смен?",
                "decision": "reflection",
                "adds": [
                    {"kind": "previous_attempt", "content": "пытался есть меньше"},
                    {"kind": "barrier", "content": "сильный голод после поздних смен возвращает к большим ужинам"},
                ],
                "contents": [{"kind": "system_reflection", "interaction": "evaluation", "semantic_content": "Связать Desired Change, Personal Meaning, Previous Attempt и Barrier в рабочую картину."}],
            },
        ],
        "current": "Да, всё именно так. Этого понимания мне достаточно, и я хочу понять, как двигаться дальше.",
    },
    "FSB-A": {
        "intent": "recognition",
        "purpose": "Two distinct prior evaluation targets make a bare affirmative genuinely ambiguous under the ACS contract.",
        "steps": [
            {
                "user": "Вечером я ем больше и не понимаю, это голод или способ снять напряжение.",
                "reply": "Здесь две разные версии: первая — вечером преобладает физический голод. Вторая — еда прежде всего помогает снять напряжение. Какая из них вам ближе?",
                "decision": "recognition",
                "adds": [{"kind": "user_interpretation", "content": "не уверен, связан ли вечерний эпизод с голодом или снятием напряжения"}],
                "contents": [
                    {"kind": "recognition_option", "interaction": "evaluation", "semantic_content": "Версия 1: в вечернем эпизоде преобладает физический голод."},
                    {"kind": "recognition_option", "interaction": "evaluation", "semantic_content": "Версия 2: еда в вечернем эпизоде прежде всего снимает напряжение."},
                ],
            },
        ],
        "current": "Да.",
    },
}


CHALLENGER_SPECS = {
    "FSB-H": {"decision_intent": "human_discovery", "semantic_focus": "Уточнить, что потеря спокойных встреч и неловкость из-за одежды значат для пользователя лично и какую часть прежней жизни он хочет вернуть.", "refs": ["current", "kind:life_change", "kind:barrier"]},
    "FSB-M": {"decision_intent": "mechanism_discovery", "semantic_focus": "Сравнить два запомнившихся эпизода по моменту появления сильного желания поесть, чтобы различить длительный голод после пропущенного обеда и напряжение после тяжёлого разговора.", "refs": ["current", "kind:personal_meaning", "kind:previous_attempt"]},
    "FSB-R": {"decision_intent": "reflection", "semantic_focus": "Отразить единую рабочую картину: смена работы нарушила режим, изменение веса ограничило встречи и одежду, а попытка просто есть меньше сталкивается с сильным голодом после поздней смены; предложить подтвердить или поправить эту связь.", "refs": ["current", "kind:life_change", "kind:personal_meaning"]},
    "FSB-S": {"decision_intent": "stop_exploration", "semantic_focus": "Завершить дальнейшее исследование этой оси: доступный вечерний паттерн зафиксирован, а начало эпизода пользователь не наблюдал и дополнительных деталей у него нет.", "refs": ["current", "kind:experience", "kind:other_relevant_user_material"]},
    "FSB-T": {"decision_intent": "transition", "semantic_focus": "Предложить с явным согласием перейти к разговору с живым консультантом, опираясь на подтверждённую картину желаемого изменения, предыдущей попытки и барьера, не подбирая решение внутри Cognitive Core.", "refs": ["current", "kind:desired_change", "kind:previous_attempt", "kind:barrier"]},
    "FSB-A": {"decision_intent": "recognition", "semantic_focus": "Сохранить неоднозначность короткого «Да» и уточнить, относится ли оно к версии о физическом голоде, к версии о снятии напряжения или к обеим, не подтверждая ни одну автоматически.", "refs": ["current", "acs_targets"]},
}


def initial_state() -> tuple[HumanModel, ActiveConversationState | None, DialogueHistory]:
    return HumanModel(updated_at=FIXED_TIME), None, DialogueHistory()


def build_case(case_id: str) -> dict[str, Any]:
    recipe = RECIPES[case_id]
    human_model, acs, history = initial_state()
    initial = {
        "human_model": human_model.model_dump(mode="json"),
        "previous_acs": None,
        "dialogue_history": history.model_dump(mode="json"),
    }
    steps = []
    with deterministic_lifecycle(case_id):
        for turn, spec in enumerate(recipe["steps"], 1):
            user_sequence = len(history.messages) + 1
            user = message(case_id, DialogueRole.USER, spec["user"], user_sequence)
            system = message(case_id, DialogueRole.SYSTEM, spec["reply"], user_sequence + 1)
            result = make_turn_result(
                case_id=case_id,
                turn=turn,
                user_text=spec["user"],
                reply=spec["reply"],
                previous_acs=acs,
                decision_intent=spec["decision"],
                additions=spec["adds"],
                contents=spec["contents"],
            )
            applied = apply_cognitive_turn(human_model, acs, history, user, system, result)
            steps.append(
                {
                    "user_message": user.model_dump(mode="json"),
                    "system_message": system.model_dump(mode="json"),
                    "cognitive_turn_result": result.model_dump(mode="json"),
                }
            )
            human_model = applied.updated_human_model
            acs = applied.new_acs
            history = applied.updated_history
    current = message(
        case_id,
        DialogueRole.USER,
        recipe["current"],
        len(history.messages) + 1,
    )
    validate_discovery_memory(human_model, acs, history)
    frozen_input = {
        "human_model": human_model.model_dump(mode="json"),
        "previous_acs": None if acs is None else acs.model_dump(mode="json"),
        "dialogue_history": history.model_dump(mode="json"),
        "current_user_message": current.model_dump(mode="json"),
        "human_experience": {"enabled": False},
    }
    return {
        "case_id": case_id,
        "coverage_intent": recipe["intent"],
        "purpose": recipe["purpose"],
        "initial_state": initial,
        "replay_sequence": steps,
        "frozen_input": frozen_input,
        "state_sha256": sha_value(frozen_input),
    }


def resolve_challenger(case: dict[str, Any]) -> dict[str, Any]:
    case_id = case["case_id"]
    spec = CHALLENGER_SPECS[case_id]
    frozen = case["frozen_input"]
    items = list(frozen["human_model"]["items"].values())
    refs: list[str] = []
    for ref in spec["refs"]:
        if ref == "current":
            refs.append(frozen["current_user_message"]["id"])
        elif ref == "acs_targets":
            refs.extend(frozen["previous_acs"]["response_targets"].keys())
        elif ref.startswith("kind:"):
            kind = ref.split(":", 1)[1]
            matching = [item["id"] for item in items if item["kind"] == kind]
            if not matching:
                raise AssertionError(f"{case_id}: no frozen item for {ref}")
            refs.append(matching[-1])
        else:
            raise AssertionError(ref)
    challenger = {
        "decision_intent": spec["decision_intent"],
        "semantic_focus": spec["semantic_focus"],
        "basis_evidence_refs": refs,
    }
    return {
        "case_id": case_id,
        "challenger": challenger,
        "challenger_sha256": sha_value(challenger),
    }


def replay_serialized(case: dict[str, Any]) -> dict[str, Any]:
    initial = case["initial_state"]
    human_model = HumanModel.model_validate(initial["human_model"])
    acs = None
    history = DialogueHistory.model_validate(initial["dialogue_history"])
    with deterministic_lifecycle(case["case_id"]):
        for step in case["replay_sequence"]:
            user = DialogueMessage.model_validate(step["user_message"])
            system = DialogueMessage.model_validate(step["system_message"])
            result = CognitiveTurnResult.model_validate(step["cognitive_turn_result"])
            applied = apply_cognitive_turn(human_model, acs, history, user, system, result)
            human_model = applied.updated_human_model
            acs = applied.new_acs
            history = applied.updated_history
    current = DialogueMessage.model_validate(case["frozen_input"]["current_user_message"])
    validate_discovery_memory(human_model, acs, history)
    if current.role != DialogueRole.USER:
        raise AssertionError(f"{case['case_id']}: current message is not USER")
    expected_sequence = history.messages[-1].sequence + 1 if history.messages else 1
    if current.sequence != expected_sequence:
        raise AssertionError(f"{case['case_id']}: current message sequence is not next")
    return {
        "human_model": human_model.model_dump(mode="json"),
        "previous_acs": None if acs is None else acs.model_dump(mode="json"),
        "dialogue_history": history.model_dump(mode="json"),
        "current_user_message": current.model_dump(mode="json"),
        "human_experience": {"enabled": False},
    }


def known_refs(frozen: dict[str, Any]) -> set[str]:
    refs = {frozen["current_user_message"]["id"]}
    refs.update(message["id"] for message in frozen["dialogue_history"]["messages"])
    refs.update(frozen["human_model"]["items"])
    refs.update(frozen["human_model"]["relations"])
    acs = frozen["previous_acs"]
    if acs:
        refs.add(acs["id"])
        refs.update(acs["active_content"])
        refs.update(acs["response_targets"])
    return refs


def validate_ambiguity_probe(case: dict[str, Any]) -> None:
    """Prove that FSB-A's short reply is expressible as AMBIGUOUS end to end."""

    frozen = case["frozen_input"]
    human_model = HumanModel.model_validate(frozen["human_model"])
    acs = ActiveConversationState.model_validate(frozen["previous_acs"])
    history = DialogueHistory.model_validate(frozen["dialogue_history"])
    user = DialogueMessage.model_validate(frozen["current_user_message"])
    reply = "Уточню: ваше «да» относится к версии о голоде, к версии о напряжении или к обеим?"
    system = message(case["case_id"], DialogueRole.SYSTEM, reply, user.sequence + 1)
    system_action = action(
        case["case_id"],
        99,
        [
            {
                "kind": "recognition_option",
                "interaction": "evaluation",
                "semantic_content": "Уточнить, к какой из двух предыдущих версий относится короткое подтверждение.",
            }
        ],
    )
    probe = CognitiveTurnResult(
        decision_intent=DecisionIntent.RECOGNITION,
        reconciliation=(
            TargetResolution(
                previous_response_target_ids=tuple(acs.response_targets),
                outcome=ReconciliationOutcome.AMBIGUOUS,
                source_span=SourceSpan(char_start=0, char_end=len(user.text)),
            ),
        ),
        system_action=system_action,
        reply_segments=(
            ReplySegment(
                local_id="segment:ambiguity_probe",
                text=reply,
                realizes_action_content_ids=tuple(
                    content.local_id for content in system_action.contents
                ),
            ),
        ),
    )
    with deterministic_lifecycle("FSB-A-ambiguity-probe"):
        applied = apply_cognitive_turn(human_model, acs, history, user, system, probe)
    validate_discovery_memory(
        applied.updated_human_model,
        applied.new_acs,
        applied.updated_history,
    )


def validate_package() -> dict[str, Any]:
    cases_doc = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    challengers_doc = json.loads(CHALLENGERS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["head"] != HEAD:
        raise AssertionError("Manifest HEAD changed")
    contract_boundary = manifest["contract_boundary"]
    if contract_boundary["cognitive_turn_schema_sha256"] != sha_value(
        CognitiveTurnResult.model_json_schema()
    ):
        raise AssertionError("CognitiveTurnResult contract schema changed")
    if contract_boundary["cognitive_core_wire_schema_sha256"] != sha_value(
        COGNITIVE_TURN_JSON_SCHEMA
    ):
        raise AssertionError("CognitiveCore wire schema changed")
    if COGNITIVE_TURN_JSON_SCHEMA.get("type") != "object":
        raise AssertionError("CognitiveCore wire root is not an object")
    selected_action = COGNITIVE_TURN_JSON_SCHEMA.get("properties", {}).get(
        "selected_action", {}
    )
    if len(selected_action.get("anyOf", ())) != len(DecisionIntent):
        raise AssertionError("CognitiveCore wire decision branches changed")
    if tuple(case["case_id"] for case in cases_doc["cases"]) != CASE_ORDER:
        raise AssertionError("Frozen case order changed.")
    challenger_by_case = {
        entry["case_id"]: entry for entry in challengers_doc["challengers"]
    }
    replay_hashes = {}
    for case in cases_doc["cases"]:
        replayed = replay_serialized(case)
        if canonical(replayed) != canonical(case["frozen_input"]):
            raise AssertionError(f"{case['case_id']}: replay is not exact")
        replay_hash = sha_value(replayed)
        if replay_hash != case["state_sha256"]:
            raise AssertionError(f"{case['case_id']}: state hash mismatch")
        if manifest["case_sha256"].get(case["case_id"]) != sha_value(case):
            raise AssertionError(f"{case['case_id']}: case hash mismatch")
        if manifest["state_sha256"].get(case["case_id"]) != replay_hash:
            raise AssertionError(f"{case['case_id']}: manifest state hash mismatch")
        if replay_hash in LEGACY_STATE_HASHES:
            raise AssertionError(f"{case['case_id']}: reuses a legacy T6/T12/T14 state")
        replay_hashes[case["case_id"]] = replay_hash
        frozen_text = canonical(case["frozen_input"])
        for forbidden in (
            "considered_possibilities",
            "candidate_list",
            "candidate_ranking",
            "chain_of_thought",
            "selected_action",
            "primary_content",
            "additional_contents",
        ):
            if forbidden in frozen_text:
                raise AssertionError(f"{case['case_id']}: production state contains {forbidden}")
        entry = challenger_by_case[case["case_id"]]
        challenger = entry["challenger"]
        if set(challenger) != {"decision_intent", "semantic_focus", "basis_evidence_refs"}:
            raise AssertionError(f"{case['case_id']}: challenger fields changed")
        if challenger["decision_intent"] != case["coverage_intent"]:
            raise AssertionError(f"{case['case_id']}: challenger intent changed")
        if not challenger["semantic_focus"].strip():
            raise AssertionError(f"{case['case_id']}: empty challenger action")
        if not challenger["basis_evidence_refs"]:
            raise AssertionError(f"{case['case_id']}: ungrounded challenger")
        unknown = set(challenger["basis_evidence_refs"]) - known_refs(case["frozen_input"])
        if unknown:
            raise AssertionError(f"{case['case_id']}: unknown challenger refs {unknown}")
        if sha_value(challenger) != entry["challenger_sha256"]:
            raise AssertionError(f"{case['case_id']}: challenger hash mismatch")
        if manifest["challenger_sha256"].get(case["case_id"]) != entry[
            "challenger_sha256"
        ]:
            raise AssertionError(f"{case['case_id']}: manifest challenger hash mismatch")
    acs_a = cases_doc["cases"][-1]["frozen_input"]["previous_acs"]
    if acs_a is None or len(acs_a["response_targets"]) < 2:
        raise AssertionError("FSB-A does not contain multiple ACS response targets")
    target_contents = {
        target["active_content_id"] for target in acs_a["response_targets"].values()
    }
    if len(target_contents) < 2:
        raise AssertionError("FSB-A targets are not materially distinct ACS content")
    validate_ambiguity_probe(cases_doc["cases"][-1])
    if manifest["api_budget"] != {
        "baseline_production": 6,
        "baseline_evaluator": 18,
        "conditional_confirmation_per_flagged_case": 4,
        "maximum_production": 12,
        "maximum_evaluator": 36,
        "maximum_total": 48,
        "retry": 0,
    }:
        raise AssertionError("Call accounting changed")
    if manifest["production_model"] != "gpt-5-mini" or manifest["evaluator_model"] != "gpt-6-astra":
        raise AssertionError("Frozen model changed")
    for name, expected in manifest["artifact_sha256"].items():
        if sha_file(PACKAGE_DIR / name) != expected:
            raise AssertionError(f"Artifact SHA mismatch: {name}")
    old_cases = json.loads(
        (OLD_PACKAGE_DIR / "fsb_01_cases_0b8366a.json").read_text(encoding="utf-8")
    )
    old_challengers = json.loads(
        (OLD_PACKAGE_DIR / "fsb_01_challengers_0b8366a.json").read_text(
            encoding="utf-8"
        )
    )
    if cases_doc["cases"] != old_cases["cases"]:
        raise AssertionError("Behavioral cases changed from the original freeze")
    if challengers_doc["challengers"] != old_challengers["challengers"]:
        raise AssertionError("Challengers changed from the original freeze")
    if PLAN_PATH.read_bytes() != (
        OLD_PACKAGE_DIR / "fsb_01_measurement_plan_0b8366a.md"
    ).read_bytes():
        raise AssertionError("Measurement policy changed from the original freeze")
    if PROMPT_PATH.read_bytes() != (
        OLD_PACKAGE_DIR / "fsb_01_evaluator_prompt_0b8366a.txt"
    ).read_bytes():
        raise AssertionError("Evaluator prompt changed from the original freeze")
    if json.loads(SCHEMA_PATH.read_text(encoding="utf-8")) != json.loads(
        (OLD_PACKAGE_DIR / "fsb_01_evaluator_schema_0b8366a.json").read_text(
            encoding="utf-8"
        )
    ):
        raise AssertionError("Evaluator schema changed from the original freeze")
    return {
        "cases_valid": len(cases_doc["cases"]),
        "replay_exact": len(replay_hashes),
        "challengers_grounded": len(challenger_by_case),
        "fsb_a_response_targets": len(acs_a["response_targets"]),
        "fsb_a_ambiguous_reconciliation_probe": "PASS",
        "state_hashes": replay_hashes,
        "behavioral_case_changed": "NO",
        "challenger_changed": "NO",
        "measurement_policy_changed": "NO",
        "only_technical_freeze_metadata_changed": "YES",
        "result": "READY FOR LIVE RUN",
    }


EVALUATOR_PROMPT = """You are a lab-only blind semantic comparator. Given one frozen Cognitive Discovery state and two anonymous proposed next semantic actions, judge which action has higher current value as the next Decision. Compare useful working-picture change, next-Decision discrimination, Discovery sufficiency, evidence availability, Product Boundary, and user cost. Do not prefer a question merely because it can collect detail. Do not continue an exhausted or unavailable line. Transition requires a sufficient working picture; stopping is legitimate. Human and mechanism discovery have no quota or mandatory order. Return only the supplied Structured Output verdict. Do not output rationale, scores, family labels, condition names, hypotheses, expected winners, or chain-of-thought.
"""


EVALUATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": [
                "LEFT_HIGHER",
                "RIGHT_HIGHER",
                "EQUIVALENT_OR_BOTH_ACCEPTABLE",
                "INSUFFICIENT_BASIS",
            ],
        }
    },
    "required": ["verdict"],
    "additionalProperties": False,
}


MEASUREMENT_PLAN = """# FSB-01 frozen measurement plan

## Question

Does the current selected-only Cognitive Core reproducibly select an observable action that is lower-value than an independently frozen eligible challenger?

The observable defect is **MISSED HIGH-VALUE ALTERNATIVE**. No inference about latent consideration, hidden cognition, candidate ranking, or chain-of-thought is permitted.

## Execution

1. Verify HEAD and every frozen SHA. A mismatch stops the run with zero API calls.
2. Run one production `CognitiveCore.propose()` call for each case in frozen baseline order, using `gpt-5-mini`, retry 0. Finish all six production calls before evaluator execution.
3. Compare every completed selected action post-hoc with its frozen challenger using three blind `gpt-6-astra` judgments and the frozen orientations.
4. If and only if a baseline aggregate is `CHALLENGER_HIGHER`, run one additional production call on the exact same frozen state, then three comparator judgments with the frozen confirmation orientations.
5. Never pass evaluator output back to production. Never modify a challenger after freeze.

## Aggregation

Use `Conversation_Lab.selected_action_evaluator.aggregate_judgments` unchanged: unanimous verdict is stable; 2/3 matching directional judgments form a directional majority; substantial disagreement or uncertainty aggregates to ABSTAIN.

## Frozen interpretation

- **CURRENT POLICY ACCEPTABLE**: all six baseline production results are valid, no aggregate is CHALLENGER_HIGHER, no critical comparison is ABSTAIN, and declared intent is semantically aligned with the selected action.
- **POLICY CHANGE JUSTIFIED**: at least one case is CHALLENGER_HIGHER both at baseline and its single conditional confirmation, with an eligible grounded challenger and no technical or measurement failure.
- **INCONCLUSIVE**: any isolated non-reproduced miss, critical ABSTAIN, invalid production/evaluator output, invalid challenger, insufficient context, or result not attributable to one concrete Decision-policy boundary.

Exact DecisionIntent family is coverage metadata, not the primary gold criterion. Pairwise semantic value against the independently frozen challenger is primary.
"""


def current_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_freeze() -> None:
    if current_head() != HEAD:
        raise SystemExit(f"HEAD mismatch: expected {HEAD}, got {current_head()}")
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    cases = [build_case(case_id) for case_id in CASE_ORDER]
    challengers = [resolve_challenger(case) for case in cases]
    write_json(CASES_PATH, {"head": HEAD, "cases": cases})
    write_json(
        CHALLENGERS_PATH,
        {
            "head": HEAD,
            "freeze_timing": "All challengers frozen offline before every future production winner.",
            "challengers": challengers,
        },
    )
    # Preserve the exact previously live-validated comparator prompt bytes.
    PROMPT_PATH.write_text(EVALUATOR_PROMPT.rstrip("\n"), encoding="utf-8")
    write_json(SCHEMA_PATH, EVALUATOR_SCHEMA)
    PLAN_PATH.write_text(MEASUREMENT_PLAN, encoding="utf-8")
    artifact_paths = [CASES_PATH, CHALLENGERS_PATH, PLAN_PATH, PROMPT_PATH, SCHEMA_PATH, Path(__file__)]
    orientations = {
        case_id: {
            "baseline": [True, False, True] if index % 2 == 0 else [False, True, False],
            "confirmation": [False, True, False] if index % 2 == 0 else [True, False, True],
        }
        for index, case_id in enumerate(CASE_ORDER)
    }
    manifest = {
        "experiment": "FSB-01 FULL SELECTED-ONLY BEHAVIORAL REGRESSION",
        "head": HEAD,
        "production_boundary": "SELECTED_ONLY",
        "production_model": "gpt-5-mini",
        "evaluator_model": "gpt-6-astra",
        "challenger_freeze": "offline_before_all_future_production_winners",
        "contract_boundary": {
            "cognitive_turn_schema_sha256": sha_value(
                CognitiveTurnResult.model_json_schema()
            ),
            "cognitive_core_wire_schema_sha256": sha_value(
                COGNITIVE_TURN_JSON_SCHEMA
            ),
            "cognitive_turn_file_sha256": sha_file(ROOT / "ai" / "cognitive_turn.py"),
            "cognitive_core_file_sha256": sha_file(ROOT / "ai" / "cognitive_core.py"),
            "production_prompt_sha256": sha_file(
                ROOT / "ai" / "prompts" / "cognitive_core_system_prompt.txt"
            ),
            "wire_root_type": COGNITIVE_TURN_JSON_SCHEMA["type"],
            "wire_root_properties": list(
                COGNITIVE_TURN_JSON_SCHEMA["properties"]
            ),
            "selected_action_branch_count": len(
                COGNITIVE_TURN_JSON_SCHEMA["properties"]["selected_action"][
                    "anyOf"
                ]
            ),
        },
        "case_sha256": {case["case_id"]: sha_value(case) for case in cases},
        "state_sha256": {
            case["case_id"]: case["state_sha256"] for case in cases
        },
        "challenger_sha256": {
            entry["case_id"]: entry["challenger_sha256"]
            for entry in challengers
        },
        "case_order": list(CASE_ORDER),
        "baseline_production_order": list(CASE_ORDER),
        "baseline_evaluator_order": list(CASE_ORDER),
        "conditional_confirmation_order": list(CASE_ORDER),
        "evaluator_orientations_selected_on_left": orientations,
        "conditional_confirmation_rule": "exactly_one_additional_production_sample_and_three_blind_judgments_for_each_baseline_CHALLENGER_HIGHER_case",
        "api_budget": {
            "baseline_production": 6,
            "baseline_evaluator": 18,
            "conditional_confirmation_per_flagged_case": 4,
            "maximum_production": 12,
            "maximum_evaluator": 36,
            "maximum_total": 48,
            "retry": 0,
        },
        "decision_rules": {
            "PASS": "CURRENT_POLICY_ACCEPTABLE",
            "FAIL": "POLICY_CHANGE_JUSTIFIED",
            "other": "INCONCLUSIVE",
        },
        "stop_rules": [
            "preflight_head_or_hash_or_offline_validation_failure_stops_with_zero_api_calls",
            "transport_billing_structural_order_or_mapping_failure_stops_without_retry",
            "no_unfrozen_prompt_contract_challenger_or_policy_mutation",
            "confirmation_only_after_baseline_CHALLENGER_HIGHER",
        ],
        "artifact_sha256": {path.name: sha_file(path) for path in artifact_paths},
    }
    write_json(MANIFEST_PATH, manifest)
    validation = validate_package()
    report_lines = [
        "# FSB-01 offline refreeze report",
        "",
        f"Result: **{validation['result']}**",
        "",
        "All six decision states are exact deterministic replays through the production Discovery contracts and `apply_cognitive_turn`. No LLM or API was used. Behavioral cases, challengers, evaluator contract and measurement policy are unchanged from the 0b8366a freeze.",
        "",
        "## Reachability",
        "",
    ]
    for case in cases:
        report_lines.append(
            f"- {case['case_id']}: {len(case['replay_sequence'])} applied turn(s); state SHA-256 `{case['state_sha256']}`."
        )
    report_lines.extend(
        [
            "",
            "FSB-A replay produces two distinct persistent ACS response targets owned by two distinct active contents. The current `Да.` can therefore be reconciled as AMBIGUOUS without inventing an unsupported ACS shape.",
            "",
            "## Frozen safeguards",
            "",
            "- Challengers contain only decision_intent, semantic_focus and basis_evidence_refs.",
            "- Every evidence reference resolves in the corresponding frozen input.",
            "- Challengers were generated before any future production winner and are hashed.",
            "- No T6/T12/T14 state hash is reused.",
            "- No candidate/consideration/evaluator data enters production state.",
            "- Baseline is 24 calls; maximum is 48; retry is zero.",
            f"- Production HEAD: `{HEAD}`.",
            f"- CognitiveTurnResult schema SHA-256: `{manifest['contract_boundary']['cognitive_turn_schema_sha256']}`.",
            f"- CognitiveCore wire schema SHA-256: `{manifest['contract_boundary']['cognitive_core_wire_schema_sha256']}`.",
            "- The wire-only selected_action representation is absent from HumanModel, ACS and DialogueHistory snapshots.",
            "",
            "## Artifact SHA-256",
            "",
        ]
    )
    for name, digest in sorted(manifest["artifact_sha256"].items()):
        report_lines.append(f"- `{name}`: `{digest}`")
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


def main() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = "0"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        raise SystemExit(completed.returncode)
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-freeze", action="store_true")
    args = parser.parse_args()
    if args.write_freeze:
        write_freeze()
    else:
        print(json.dumps(validate_package(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
