"""M1a: разбор транскрипта, псевдонимизация, фильтр шаблонов, сборка реплик.

Формат входа: UTF-8 (возможна BOM). Первая строка — заголовок.
Далее записи `ЧЧ:ММ:СС<TAB>Метка говорящего<TAB>Текст`, одна на строку.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from qa.settings import PipelineSettings

logger = logging.getLogger(__name__)

ROLES = frozenset({"INTERVIEWER", "RESPONDENT", "OBSERVER"})

_LINE_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\t([^\t]+)\t(.*)$")

# Фильтр шаблонов PII (CONSTITUTION §5): совпадения помечаются, не удаляются
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # e-mail
    re.compile(r"(?:\+7|8)[\s(]*\d{3}[\s)]*\d{3}[\s-]*\d{2}[\s-]*\d{2}"),  # телефон
    re.compile(r"\d[\d\s-]{12,}\d"),  # длинные числа (похожи на номера счетов/карт)
)


class TranscriptParseError(Exception):
    """Ошибка разбора транскрипта с номером строки."""


class TranscriptMetaError(Exception):
    """Ошибка в interview.yaml."""


@dataclass(frozen=True)
class TranscriptRecord:
    """Одна запись транскрипта (строка)."""

    line_no: int
    start_sec: int
    speaker_label: str
    text: str


@dataclass(frozen=True)
class InterviewMeta:
    """Метаданные интервью из interview.yaml."""

    interview_id: str
    respondent_code: str
    group: str
    roles: dict[str, str]
    names_to_codes: dict[str, str]
    date: str
    transcript_source: str


@dataclass
class UtteranceDraft:
    """Сборная реплика до записи в БД."""

    speaker_role: str
    parts: list[str] = field(default_factory=list)
    turn_ids: list[int] = field(default_factory=list)
    is_backchannel: bool = False

    @property
    def text(self) -> str:
        return " ".join(self.parts)

    @property
    def order_key(self) -> int:
        """Позиция в интервью: id первой входящей записи."""
        return self.turn_ids[0]


def parse_transcript(path: Path) -> tuple[str, list[TranscriptRecord]]:
    """Разобрать файл транскрипта.

    Returns:
        (заголовок_первой_строки, записи).

    Raises:
        TranscriptParseError: строка не по формату или метка времени убывает
            (с номером строки; молча пропускать запрещено).
    """
    content = path.read_text(encoding="utf-8-sig")
    lines = content.splitlines()
    if not lines:
        msg = "Файл транскрипта пуст"
        raise TranscriptParseError(msg)

    header = lines[0].strip()
    records: list[TranscriptRecord] = []
    prev_sec = 0
    for idx, line in enumerate(lines[1:], start=2):
        match = _LINE_RE.match(line)
        if match is None:
            msg = f"Строка {idx} не по формату ЧЧ:ММ:СС<TAB>Метка<TAB>Текст: {line[:60]!r}"
            raise TranscriptParseError(msg)
        hh, mm, ss, label, text = match.groups()
        h_i, m_i, s_i = int(hh), int(mm), int(ss)
        if m_i > 59 or s_i > 59:
            msg = f"Строка {idx}: некорректная временная метка {hh}:{mm}:{ss}"
            raise TranscriptParseError(msg)
        start_sec = h_i * 3600 + m_i * 60 + s_i
        if start_sec < prev_sec:
            msg = f"Строка {idx}: метка времени убывает ({start_sec} < {prev_sec})"
            raise TranscriptParseError(msg)
        prev_sec = start_sec
        records.append(
            TranscriptRecord(line_no=idx, start_sec=start_sec, speaker_label=label.strip(), text=text.strip())
        )
    return header, records


def load_meta(path: Path) -> InterviewMeta:
    """Прочитать и проверить interview.yaml.

    Raises:
        TranscriptMetaError: отсутствует поле или роль вне списка.
    """
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    required = ["interview_id", "respondent_code", "group", "roles", "date", "transcript_source"]
    missing = [k for k in required if k not in data]
    if missing:
        msg = f"В interview.yaml не хватает полей: {missing}"
        raise TranscriptMetaError(msg)
    roles: dict[str, str] = dict(data["roles"])
    bad = {label: role for label, role in roles.items() if role not in ROLES}
    if bad:
        msg = f"Недопустимые роли в interview.yaml: {bad}. Допустимы: {sorted(ROLES)}"
        raise TranscriptMetaError(msg)
    return InterviewMeta(
        interview_id=str(data["interview_id"]),
        respondent_code=str(data["respondent_code"]),
        group=str(data["group"]),
        roles=roles,
        names_to_codes=dict(data.get("names_to_codes") or {}),
        date=str(data["date"]),
        transcript_source=str(data["transcript_source"]),
    )


def build_replacement_map(meta: InterviewMeta, extra_names: dict[str, str] | None = None) -> dict[str, str]:
    """Карта «что заменяем -> на что» для псевдонимизации текста.

    Имена из names_to_codes (приоритет) плюс внешняя таблица (NAMES_MAP_PATH).
    Метки говорящих тоже заменяются: респондент — на его код, остальные — на роль.
    """
    mapping: dict[str, str] = dict(extra_names or {})
    mapping.update(meta.names_to_codes)
    for label, role in meta.roles.items():
        mapping.setdefault(label, meta.respondent_code if role == "RESPONDENT" else role)
    return mapping


def pseudonymize(text: str, replacement_map: dict[str, str]) -> str:
    """Заменить имена и метки на коды (длинные совпадения — первыми)."""
    result = text
    for surface in sorted(replacement_map, key=len, reverse=True):
        if surface:
            result = result.replace(surface, replacement_map[surface])
    return result


def detect_pii(text: str) -> bool:
    """Есть ли в тексте совпадения с шаблонами PII."""
    return any(p.search(text) for p in _PII_PATTERNS)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z]+", text.lower())


def is_backchannel(text: str, words: set[str], max_words: int) -> bool:
    """Короткая вставка-подтверждение: все слова из списка, длина в лимите."""
    tokens = _tokenize(text)
    return 0 < len(tokens) <= max_words and all(t in words for t in tokens)


def build_utterances(
    turns: list[tuple[int, str, str]],
    backchannel_words: set[str],
    backchannel_max_words: int,
) -> list[UtteranceDraft]:
    """Собрать реплики из записей.

    Args:
        turns: список (turn_id, speaker_role, text) в порядке транскрипта.
        backchannel_words: список слов-подтверждений.
        backchannel_max_words: максимум слов во вставке.

    Подряд идущие записи одного говорящего, разделённые только короткими
    вставками другого, объединяются в одну реплику. Вставки становятся
    отдельными репликами с is_backchannel=1 и не удаляются.
    """
    finished: list[UtteranceDraft] = []
    current: UtteranceDraft | None = None

    for turn_id, role, text in turns:
        if current is not None and role == current.speaker_role:
            current.parts.append(text)
            current.turn_ids.append(turn_id)
        elif current is not None and is_backchannel(text, backchannel_words, backchannel_max_words):
            # Вставка другого говорящего: своя реплика-бэкканал, текущая продолжается
            insert = UtteranceDraft(speaker_role=role, parts=[text], turn_ids=[turn_id], is_backchannel=True)
            finished.append(insert)
        else:
            if current is not None:
                finished.append(current)
            current = UtteranceDraft(speaker_role=role, parts=[text], turn_ids=[turn_id])

    if current is not None:
        finished.append(current)

    finished.sort(key=lambda u: u.order_key)
    return finished


@dataclass(frozen=True)
class LoadResult:
    """Итог загрузки интервью."""

    interview_id: int
    n_turns: int
    n_utterances: int
    n_pii_flagged: int
    skipped: bool


def load_transcript(
    conn: sqlite3.Connection,
    transcript_path: Path,
    meta: InterviewMeta,
    replacement_map: dict[str, str],
    pipeline: PipelineSettings,
) -> LoadResult:
    """Разобрать транскрипт, псевдонимизировать, пометить PII, собрать реплики, записать в БД.

    Повторная загрузка того же interview_id не создаёт дублей.
    """
    existing = conn.execute("SELECT id FROM interview WHERE id_external = ?", (meta.interview_id,)).fetchone()
    if existing is not None:
        logger.info("Интервью %s уже загружено (id=%s), пропуск", meta.interview_id, existing[0])
        return LoadResult(interview_id=existing[0], n_turns=0, n_utterances=0, n_pii_flagged=0, skipped=True)

    header, records = parse_transcript(transcript_path)

    # Проверка меток: каждая метка транскрипта должна быть описана в ролях
    unknown = {r.speaker_label for r in records} - set(meta.roles)
    if unknown:
        msg = f"Метки говорящих не описаны в interview.yaml: {sorted(unknown)}"
        raise TranscriptMetaError(msg)

    interview_id_raw = conn.execute(
        'INSERT INTO interview (id_external, name, respondent_code, "group", transcript_source) VALUES (?, ?, ?, ?, ?)',
        (meta.interview_id, header, meta.respondent_code, meta.group, meta.transcript_source),
    ).lastrowid
    if interview_id_raw is None:
        msg = "SQLite не вернул id вставленной строки interview"
        raise sqlite3.Error(msg)
    interview_id = interview_id_raw

    turns: list[tuple[int, str, str]] = []
    n_pii = 0
    for rec in records:
        role = meta.roles[rec.speaker_label]
        text = pseudonymize(rec.text, replacement_map)
        pii = detect_pii(text)
        n_pii += int(pii)
        turn_id_raw = conn.execute(
            "INSERT INTO turn (interview_id, line_no, start_sec, speaker_role, text, pii_flag)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (interview_id, rec.line_no, rec.start_sec, role, text, int(pii)),
        ).lastrowid
        if turn_id_raw is None:
            msg = f"SQLite не вернул id вставленной строки turn (строка {rec.line_no})"
            raise sqlite3.Error(msg)
        turn_id = turn_id_raw
        turns.append((turn_id, role, text))

    drafts = build_utterances(turns, set(pipeline.backchannel_words), pipeline.backchannel_max_words)
    for draft in drafts:
        conn.execute(
            "INSERT INTO utterance (interview_id, speaker_role, text, turn_ids, is_backchannel) VALUES (?, ?, ?, ?, ?)",
            (interview_id, draft.speaker_role, draft.text, json.dumps(draft.turn_ids), int(draft.is_backchannel)),
        )

    conn.commit()
    logger.info(
        "Интервью %s загружено: turns=%s, utterances=%s, pii=%s",
        meta.interview_id,
        len(turns),
        len(drafts),
        n_pii,
    )
    return LoadResult(
        interview_id=interview_id,
        n_turns=len(turns),
        n_utterances=len(drafts),
        n_pii_flagged=n_pii,
        skipped=False,
    )


def load_names_map(env_path: str) -> dict[str, str]:
    """Загрузить внешнюю таблицу «имя -> код» (NAMES_MAP_PATH)."""
    path = Path(env_path)
    if not path.exists():
        msg = f"NAMES_MAP_PATH указывает на несуществующий файл: {path}"
        raise FileNotFoundError(msg)
    with open(path, encoding="utf-8") as f:
        data: dict[str, str] = yaml.safe_load(f)
    return dict(data)
