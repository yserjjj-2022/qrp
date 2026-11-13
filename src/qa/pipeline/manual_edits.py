"""Разовые правки реплик: «в этой реплике имелось в виду Z» (M8).

Правка живёт в JSON-файле рядом с данными интервью и применяется на границе
L1 -> L2: `utterance.text` (L1) не меняется (CONSTITUTION §4), исправленный
текст видит только нормализация, поэтому модель работает с ним, а журнал
изменений фиксирует правку как отдельный тип.

Формат файла::

    {
      "edits": [
        {"utterance_id": 42, "from": "искажение", "to": "правильное слово"}
      ]
    }

Хеш файла пишется в `run.params_json` нормализации, как хеш лексикона: правка
меняет результат, и это должно быть видно в прогоне.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class ManualEditsError(Exception):
    """Файл правок не читается или нарушает формат."""


@dataclass(frozen=True)
class ManualEdit:
    """Одна правка: что заменить в конкретной реплике."""

    utterance_id: int
    source: str
    target: str


@dataclass(frozen=True)
class ManualEdits:
    """Правки интервью вместе с хешем файла."""

    path: Path | None = None
    file_hash: str = ""
    by_utterance: dict[int, tuple[ManualEdit, ...]] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Сколько правок загружено."""
        return sum(len(items) for items in self.by_utterance.values())

    def apply(self, utterance_id: int, text: str) -> tuple[str, tuple[ManualEdit, ...]]:
        """Применить правки реплики; вернуть текст и применённые записи.

        Замена идёт по точному вхождению без учёта регистра. Если исходная
        строка в реплике не встречается, правка не применяется и в журнал не
        попадает: молчаливая «правка в пустоту» скрыла бы ошибку в файле.
        """
        edits = self.by_utterance.get(utterance_id)
        if not edits:
            return text, ()
        result = text
        applied: list[ManualEdit] = []
        for edit in edits:
            replaced = _replace_ignore_case(result, edit.source, edit.target)
            if replaced == result:
                logger.warning(
                    "Разовая правка для реплики %d не применена: исходная форма не найдена",
                    utterance_id,
                )
                continue
            result = replaced
            applied.append(edit)
        return result, tuple(applied)


def _replace_ignore_case(text: str, source: str, target: str) -> str:
    """Заменить все вхождения `source` на `target` без учёта регистра."""
    if not source:
        return text
    lowered = text.casefold()
    needle = source.casefold()
    if needle not in lowered:
        return text
    parts: list[str] = []
    position = 0
    while True:
        found = lowered.find(needle, position)
        if found < 0:
            parts.append(text[position:])
            break
        parts.append(text[position:found])
        parts.append(target)
        position = found + len(needle)
    return "".join(parts)


def manual_edits_path(exports_dir: Path, interview_code: str) -> Path:
    """Путь к файлу правок интервью."""
    return Path(exports_dir) / f"quote_edits_{interview_code}.json"


def load_manual_edits(path: Path) -> ManualEdits:
    """Прочитать файл правок; отсутствие файла — норма, а не ошибка.

    Raises:
        ManualEditsError: файл есть, но не является JSON-объектом нужного вида.
    """
    if not path.is_file():
        return ManualEdits()
    raw = path.read_bytes()
    file_hash = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"Файл разовых правок {path} не является корректным JSON"
        raise ManualEditsError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"Файл разовых правок {path}: ожидался JSON-объект с ключом edits"
        raise ManualEditsError(msg)
    items = payload.get("edits")
    if items is None:
        items = []
    if not isinstance(items, list):
        msg = f"Файл разовых правок {path}: ключ edits должен быть массивом"
        raise ManualEditsError(msg)

    by_utterance: dict[int, list[ManualEdit]] = {}
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            msg = f"Файл разовых правок {path}: запись {index} не является объектом"
            raise ManualEditsError(msg)
        raw_id = item.get("utterance_id")
        try:
            utterance_id = int(str(raw_id).strip())
        except (TypeError, ValueError) as exc:
            msg = f"Файл разовых правок {path}: у записи {index} некорректный utterance_id"
            raise ManualEditsError(msg) from exc
        source = str(item.get("from") or "").strip()
        target = str(item.get("to") or "").strip()
        if not source or not target:
            msg = f"Файл разовых правок {path}: у записи {index} пустое from или to"
            raise ManualEditsError(msg)
        by_utterance.setdefault(utterance_id, []).append(ManualEdit(utterance_id, source, target))

    result = ManualEdits(
        path=path,
        file_hash=file_hash,
        by_utterance={key: tuple(value) for key, value in by_utterance.items()},
    )
    logger.info("Разовые правки %s: %d записей на %d реплик", path.name, result.count, len(result.by_utterance))
    return result


__all__ = [
    "ManualEdit",
    "ManualEdits",
    "ManualEditsError",
    "load_manual_edits",
    "manual_edits_path",
]
