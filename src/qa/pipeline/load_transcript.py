"""M1a: разбор транскрипта, псевдонимизация, фильтр шаблонов, сборка реплик.

Формат входа: UTF-8 (возможна BOM). Первая строка — заголовок.
Далее записи `ЧЧ:ММ:СС<TAB>Метка говорящего<TAB>Текст`, одна на строку.

Код респондента берётся из имени файла (R01.txt), дата — из заголовка.
Роли говорящих определяются по числу слов и подтверждаются оператором.
Заголовок интервью (interview.name) тоже проходит псевдонимизацию.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from qa.settings import GroupsSettings, PipelineSettings, normalize_group

logger = logging.getLogger(__name__)

ROLES = frozenset({"INTERVIEWER", "RESPONDENT", "OBSERVER"})

_LINE_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\t([^\t]+)\t(.*)$")
_RESPONDENT_CODE_RE = re.compile(r"^R\d+$")
_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DATE_DOTTED_RE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")

# Фильтр шаблонов PII (CONSTITUTION §5): совпадения помечаются, не удаляются.
# pii_flag только помечает: загрузка и отправка в модель не блокируются.
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # e-mail
    re.compile(r"(?:\+7|8)[\s(]*\d{3}[\s)]*\d{3}[\s-]*\d{2}[\s-]*\d{2}"),  # телефон
    re.compile(r"\d[\d\s-]{12,}\d"),  # длинные числа (похожи на номера счетов/карт)
)


class TranscriptParseError(Exception):
    """Ошибка разбора транскрипта с номером строки."""


class TranscriptMetaError(Exception):
    """Ошибка метаданных интервью."""


@dataclass(frozen=True)
class TranscriptRecord:
    """Одна запись транскрипта (строка)."""

    line_no: int
    start_sec: int
    speaker_label: str
    text: str


@dataclass(frozen=True)
class InterviewMeta:
    """Метаданные интервью (из листа «Интервью» или вычисленные по транскрипту)."""

    interview_id: str
    respondent_code: str
    group: str
    roles: dict[str, str]
    names_to_codes: dict[str, str]
    date: str | None
    transcript_source: str | None


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
    if _LINE_RE.match(lines[0]):
        msg = "Строка 1: отсутствует заголовок транскрипта (первая строка уже является записью)"
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


def parse_header_date(header: str) -> str | None:
    """Дата интервью из заголовка: ISO или ДД.ММ.ГГГГ.

    Raises:
        TranscriptMetaError: дата выглядит некорректной.
    """
    iso = _DATE_RE.search(header)
    if iso is not None:
        year, month, day = (int(part) for part in iso.groups())
        try:
            return date(year, month, day).isoformat()
        except ValueError as exc:
            msg = f"Некорректная дата в заголовке транскрипта: {iso.group(0)!r}"
            raise TranscriptMetaError(msg) from exc
    dotted = _DATE_DOTTED_RE.search(header)
    if dotted is not None:
        day, month, year = (int(part) for part in dotted.groups())
        try:
            return date(year, month, day).isoformat()
        except ValueError as exc:
            msg = f"Некорректная дата в заголовке транскрипта: {dotted.group(0)!r}"
            raise TranscriptMetaError(msg) from exc
    return None


def respondent_code_from_filename(path: Path) -> str:
    """Код респондента из имени файла: R01.txt -> R01.

    Raises:
        TranscriptMetaError: имя файла не содержит кода вида R01.
    """
    stem = path.stem.strip()
    if not _RESPONDENT_CODE_RE.match(stem):
        msg = f"Имя файла {path.name!r} не содержит кода респондента вида R01. Переименуйте файл по коду респондента"
        raise TranscriptMetaError(msg)
    return stem


def count_speaker_words(records: list[TranscriptRecord]) -> dict[str, int]:
    """Число слов по меткам говорящих (в порядке первого появления)."""
    counts: dict[str, int] = {}
    for record in records:
        counts.setdefault(record.speaker_label, 0)
        counts[record.speaker_label] += len(re.findall(r"[\w-]+", record.text, flags=re.UNICODE))
    return counts


def propose_roles(word_counts: dict[str, int]) -> dict[str, str]:
    """Предложить роли: больше всего слов — респондент, второй — интервьюер.

    Остальные говорящие получают роль наблюдателя. При равенстве слов порядок
    появления в транскрипте сохраняется.
    """
    ordered = sorted(word_counts.items(), key=lambda item: (-item[1], list(word_counts).index(item[0])))
    roles: dict[str, str] = {}
    for index, (label, _words) in enumerate(ordered):
        if index == 0:
            roles[label] = "RESPONDENT"
        elif index == 1:
            roles[label] = "INTERVIEWER"
        else:
            roles[label] = "OBSERVER"
    return roles


def format_roles_table(word_counts: dict[str, int], roles: dict[str, str]) -> str:
    """Таблица предложенных ролей для показа оператору."""
    lines = ["Предложенные роли говорящих:", "  метка\tслов\tроль"]
    for label, words in word_counts.items():
        lines.append(f"  {label}\t{words}\t{roles.get(label, '?')}")
    return "\n".join(lines)


def load_meta(path: Path) -> InterviewMeta:
    """Прочитать и проверить устаревший interview.yaml (совместимость).

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
    """Есть ли в тексте совпадения с шаблонами PII (только пометка)."""
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
            # Вставка другого говорящего: своя реплика-бэкканал, текущая продолжается.
            # Подряд идущие вставки одного говорящего объединяются между собой.
            last = finished[-1] if finished else None
            if last is not None and last.is_backchannel and last.speaker_role == role:
                last.parts.append(text)
                last.turn_ids.append(turn_id)
            else:
                finished.append(
                    UtteranceDraft(speaker_role=role, parts=[text], turn_ids=[turn_id], is_backchannel=True)
                )
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
    *,
    groups: GroupsSettings | None = None,
) -> LoadResult:
    """Разобрать транскрипт, псевдонимизировать, пометить PII, собрать реплики, записать в БД.

    Заголовок интервью тоже проходит псевдонимизацию. Пропуск только если у интервью
    уже есть записи turn. Если строка interview создана раньше (init-interview или лист
    «Интервью») и записей ещё нет, строка дополняется, а не дублируется.

    Args:
        groups: канонические значения групп для нормализации группы интервью.

    Raises:
        TranscriptParseError: файл не по формату.
        TranscriptMetaError: метки говорящих не описаны в ролях интервью.
    """
    existing = conn.execute(
        "SELECT i.id, (SELECT COUNT(*) FROM turn t WHERE t.interview_id = i.id)"
        " FROM interview i WHERE i.id_external = ?",
        (meta.interview_id,),
    ).fetchone()
    if existing is not None and int(existing[1]) > 0:
        logger.info("Интервью %s уже загружено (id=%s), пропуск", meta.interview_id, existing[0])
        return LoadResult(interview_id=existing[0], n_turns=0, n_utterances=0, n_pii_flagged=0, skipped=True)

    header, records = parse_transcript(transcript_path)

    unknown = {r.speaker_label for r in records} - set(meta.roles)
    if unknown:
        msg = f"Метки говорящих не описаны в ролях интервью: {sorted(unknown)}"
        raise TranscriptMetaError(msg)

    interview_date = meta.date or parse_header_date(header)
    safe_header = pseudonymize(header, replacement_map)
    group = normalize_group(meta.group, groups)

    if existing is None:
        interview_id_raw = conn.execute(
            'INSERT INTO interview (id_external, name, respondent_code, "group", transcript_source, interview_date)'
            " VALUES (?, ?, ?, ?, ?, ?)",
            (meta.interview_id, safe_header, meta.respondent_code, group, meta.transcript_source, interview_date),
        ).lastrowid
        if interview_id_raw is None:
            msg = "SQLite не вернул id вставленной строки interview"
            raise sqlite3.Error(msg)
        interview_id = int(interview_id_raw)
    else:
        interview_id = int(existing[0])
        conn.execute(
            'UPDATE interview SET name = ?, respondent_code = ?, "group" = ?,'
            " transcript_source = COALESCE(?, transcript_source),"
            " interview_date = COALESCE(?, interview_date) WHERE id = ?",
            (safe_header, meta.respondent_code, group, meta.transcript_source, interview_date, interview_id),
        )
        logger.info("Интервью %s дополнено записями транскрипта (id=%s)", meta.interview_id, interview_id)

    for label, role in meta.roles.items():
        words = sum(len(re.findall(r"[\w-]+", r.text, flags=re.UNICODE)) for r in records if r.speaker_label == label)
        conn.execute(
            "INSERT INTO interview_speaker (interview_id, speaker_label, role, words) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(interview_id, speaker_label) DO UPDATE SET role = excluded.role, words = excluded.words",
            (interview_id, label, role, words),
        )

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
        turn_id = int(turn_id_raw)
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
        data: dict[str, str] = yaml.safe_load(f) or {}
    return {str(key): str(value) for key, value in dict(data).items()}


def ensure_names_map_skeleton(env_path: str, labels: list[str]) -> Path:
    """Создать заготовку таблицы «имя -> код» вне проекта, если файла нет.

    Существующий файл не перезаписывается: дописываются только новые метки.
    """
    path = Path(env_path)
    data: dict[str, str] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = {str(key): str(value) for key, value in (yaml.safe_load(f) or {}).items()}
    changed = False
    for label in labels:
        if label not in data:
            data[label] = ""
            changed = True
    if changed or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=True)
    return path
