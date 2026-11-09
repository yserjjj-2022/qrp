"""Репозитории рабочей книги: версии, кодбук, темы, лексикон, факты, интервью.

Весь SQL по этим сущностям живёт здесь; пайплайн и CLI его не содержат.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any

from qa.pipeline.workbook import WorkbookData
from qa.settings import GroupsSettings, resolve_group

logger = logging.getLogger(__name__)


def _group_value(value: Any, groups: GroupsSettings | None) -> Any:
    """Каноническая группа; неизвестное значение сохраняется как есть с предупреждением."""
    if value is None or groups is None:
        return value
    canonical, recognized = resolve_group(str(value), groups)
    if not recognized:
        logger.warning("Неизвестная группа %r: сохранена как есть", str(value))
    return canonical


@dataclass(frozen=True)
class CodebookVersionRef:
    """Ссылка на версию кодбука и версию книги."""

    codebook_version_id: int
    workbook_version_id: int
    created: bool


class WorkbookRepository:
    """Версии книги и хеши листов."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def find_by_hash(self, file_hash: str) -> int | None:
        """id версии книги по хешу файла."""
        row = self._conn.execute("SELECT id FROM workbook_version WHERE file_hash = ?", (file_hash,)).fetchone()
        return int(row[0]) if row is not None else None

    def insert(self, workbook: WorkbookData) -> int:
        """Записать версию книги и хеши листов."""
        cursor = self._conn.execute(
            "INSERT INTO workbook_version (workbook_path, file_hash, codebook_hash) VALUES (?, ?, ?)",
            (str(workbook.path), workbook.file_hash, workbook.codebook_hash),
        )
        workbook_version_id = cursor.lastrowid
        if workbook_version_id is None:
            msg = "SQLite не вернул id вставленной строки workbook_version"
            raise sqlite3.Error(msg)
        for name, data in workbook.sheets.items():
            self._conn.execute(
                "INSERT INTO workbook_sheet (workbook_version_id, sheet_name, header_hash, content_hash, n_rows)"
                " VALUES (?, ?, ?, ?, ?)",
                (workbook_version_id, name, data.header_hash, data.content_hash, len(data.rows)),
            )
        return int(workbook_version_id)

    def sheet_hashes(self, workbook_version_id: int) -> dict[str, str]:
        """Хеши листов записанной версии."""
        rows = self._conn.execute(
            "SELECT sheet_name, content_hash FROM workbook_sheet WHERE workbook_version_id = ?",
            (workbook_version_id,),
        ).fetchall()
        return {str(name): str(value) for name, value in rows}


class CodebookRepository:
    """Версии кодбука, коды, вопросы, атрибуты, связи вопрос-код."""

    def __init__(self, conn: sqlite3.Connection, groups: GroupsSettings | None = None) -> None:
        self._conn = conn
        self._groups = groups

    def find_by_codebook_hash(self, codebook_hash: str) -> int | None:
        """id версии кодбука по хешу кодбуковых листов."""
        row = self._conn.execute("SELECT id FROM codebook_version WHERE codebook_hash = ?", (codebook_hash,)).fetchone()
        return int(row[0]) if row is not None else None

    def insert_version(self, workbook_version_id: int, codebook_hash: str) -> int:
        """Создать версию кодбука для версии книги."""
        cursor = self._conn.execute(
            "INSERT INTO codebook_version (file_hash, workbook_version_id, codebook_hash) VALUES (?, ?, ?)",
            (codebook_hash, workbook_version_id, codebook_hash),
        )
        version_id = cursor.lastrowid
        if version_id is None:
            msg = "SQLite не вернул id вставленной строки codebook_version"
            raise sqlite3.Error(msg)
        return int(version_id)

    def insert_code(self, version_id: int, row: dict[str, Any]) -> int:
        """Вставить код из строки листа «Коды»."""
        cursor = self._conn.execute(
            "INSERT INTO code (codebook_version_id, family, name, title, kind, definition, include, exclude,"
            " tiebreak_rule, origin, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'base', 'active')",
            (
                version_id,
                row.get("family"),
                row.get("code"),
                row.get("title"),
                row.get("kind"),
                row.get("definition"),
                row.get("include"),
                row.get("exclude"),
                row.get("tiebreak_rule"),
            ),
        )
        code_id = cursor.lastrowid
        if code_id is None:
            msg = "SQLite не вернул id вставленной строки code"
            raise sqlite3.Error(msg)
        return int(code_id)

    def insert_question(self, version_id: int, row: dict[str, Any]) -> int:
        """Вставить вопрос гайда."""
        cursor = self._conn.execute(
            'INSERT INTO question (codebook_version_id, identifier, block, "group", wording, goal, notes)'
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                version_id,
                row.get("question_id"),
                row.get("block"),
                _group_value(row.get("group"), self._groups),
                row.get("wording"),
                row.get("goal"),
                row.get("notes"),
            ),
        )
        question_id = cursor.lastrowid
        if question_id is None:
            msg = "SQLite не вернул id вставленной строки question"
            raise sqlite3.Error(msg)
        return int(question_id)

    def link_question_code(self, question_id: int, code_id: int) -> None:
        """Связать вопрос и код (идемпотентно)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO question_code (question_id, code_id) VALUES (?, ?)",
            (question_id, code_id),
        )

    def insert_attribute_definition(self, version_id: int, row: dict[str, Any]) -> None:
        """Вставить описание атрибута разметки."""
        self._conn.execute(
            "INSERT OR IGNORE INTO attribute_definition (codebook_version_id, name, kind, description)"
            " VALUES (?, ?, ?, ?)",
            (version_id, row.get("name"), row.get("kind"), row.get("description")),
        )

    def insert_glossary_term(self, version_id: int, term: str, replacement: str | None) -> None:
        """Вставить термин словаря замен."""
        self._conn.execute(
            "INSERT OR IGNORE INTO glossary_term (codebook_version_id, term, replacement) VALUES (?, ?, ?)",
            (version_id, term, replacement),
        )

    def counts(self, version_id: int) -> dict[str, int]:
        """Счётчики загруженной версии."""
        code = self._conn.execute("SELECT COUNT(*) FROM code WHERE codebook_version_id = ?", (version_id,)).fetchone()
        question = self._conn.execute(
            "SELECT COUNT(*) FROM question WHERE codebook_version_id = ?", (version_id,)
        ).fetchone()
        link = self._conn.execute(
            "SELECT COUNT(*) FROM question_code qc JOIN code c ON qc.code_id = c.id WHERE c.codebook_version_id = ?",
            (version_id,),
        ).fetchone()
        glossary = self._conn.execute(
            "SELECT COUNT(*) FROM glossary_term WHERE codebook_version_id = ?", (version_id,)
        ).fetchone()
        attributes = self._conn.execute(
            "SELECT COUNT(*) FROM attribute_definition WHERE codebook_version_id = ?", (version_id,)
        ).fetchone()
        return {
            "codes": int(code[0]),
            "questions": int(question[0]),
            "question_code": int(link[0]),
            "glossary": int(glossary[0]),
            "attributes": int(attributes[0]),
        }

    def code_ids(self, version_id: int) -> dict[str, int]:
        """Карта «код -> id» для версии."""
        rows = self._conn.execute("SELECT name, id FROM code WHERE codebook_version_id = ?", (version_id,)).fetchall()
        return {str(name): int(code_id) for name, code_id in rows}

    def question_ids(self, version_id: int) -> dict[str, int]:
        """Карта «идентификатор вопроса -> id» для версии."""
        rows = self._conn.execute(
            "SELECT identifier, id FROM question WHERE codebook_version_id = ?", (version_id,)
        ).fetchall()
        return {str(identifier): int(qid) for identifier, qid in rows}


class ThemeRepository:
    """Дерево тем: узлы, темы, привязки кодов."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def clear(self) -> None:
        """Очистить дерево тем перед пересборкой из книги."""
        self._conn.execute("DELETE FROM theme_code")
        self._conn.execute("DELETE FROM theme")

    def insert_node(self, name: str) -> int:
        """Узел верхнего уровня."""
        cursor = self._conn.execute("INSERT INTO theme (name, parent_id, level) VALUES (?, NULL, 0)", (name,))
        node_id = cursor.lastrowid
        if node_id is None:
            msg = "SQLite не вернул id вставленного узла theme"
            raise sqlite3.Error(msg)
        return int(node_id)

    def insert_theme(self, name: str, parent_id: int) -> int:
        """Тема внутри узла."""
        cursor = self._conn.execute("INSERT INTO theme (name, parent_id, level) VALUES (?, ?, 1)", (name, parent_id))
        theme_id = cursor.lastrowid
        if theme_id is None:
            msg = "SQLite не вернул id вставленной темы theme"
            raise sqlite3.Error(msg)
        return int(theme_id)

    def bind(self, theme_id: int, code_id: int) -> None:
        """Привязать код к теме."""
        self._conn.execute("INSERT OR IGNORE INTO theme_code (theme_id, code_id) VALUES (?, ?)", (theme_id, code_id))

    def tree(self) -> list[tuple[str, list[tuple[str, list[str]]]]]:
        """Дерево для показа оператору: узел -> [(тема, [коды])]."""
        nodes = self._conn.execute("SELECT id, name FROM theme WHERE parent_id IS NULL ORDER BY id").fetchall()
        result: list[tuple[str, list[tuple[str, list[str]]]]] = []
        for node_id, node_name in nodes:
            themes = self._conn.execute(
                "SELECT id, name FROM theme WHERE parent_id = ? ORDER BY id", (node_id,)
            ).fetchall()
            theme_items: list[tuple[str, list[str]]] = []
            for theme_id, theme_name in themes:
                codes = self._conn.execute(
                    "SELECT c.name FROM theme_code tc JOIN code c ON tc.code_id = c.id"
                    " WHERE tc.theme_id = ? ORDER BY c.id",
                    (theme_id,),
                ).fetchall()
                theme_items.append((str(theme_name), [str(row[0]) for row in codes]))
            result.append((str(node_name), theme_items))
        return result


class LexiconRepository:
    """Лексикон терминов; решения review-terms имеют приоритет над листом."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert_from_workbook(
        self, canonical: str, variants: list[str], definition: str | None, status: str | None
    ) -> bool:
        """Записать термин из книги, не перезаписывая решение review-terms.

        Returns:
            True, если запись вставлена или обновлена; False, если пропущена.
        """
        row = self._conn.execute(
            "SELECT source, variants_json, definition, status FROM lexicon WHERE canonical = ?", (canonical,)
        ).fetchone()
        if row is not None and row[0] == "review":
            return False
        self._conn.execute(
            "INSERT INTO lexicon (canonical, variants_json, definition, status, source)"
            " VALUES (?, ?, ?, ?, 'workbook')"
            " ON CONFLICT(canonical) DO UPDATE SET variants_json = excluded.variants_json,"
            " definition = excluded.definition, status = excluded.status,"
            " updated_at = CURRENT_TIMESTAMP",
            (canonical, json.dumps(variants, ensure_ascii=False), definition, status or "proposed"),
        )
        return True

    def all_terms(self) -> dict[str, dict[str, Any]]:
        """Все термины лексикона."""
        rows = self._conn.execute(
            "SELECT canonical, variants_json, definition, status, source FROM lexicon ORDER BY canonical"
        ).fetchall()
        return {
            str(canonical): {
                "variants": json.loads(variants_json or "[]"),
                "definition": definition,
                "status": status,
                "source": source,
            }
            for canonical, variants_json, definition, status, source in rows
        }

    def apply_review(self, canonical: str, decision: str) -> bool:
        """Зафиксировать решение оператора по термину.

        decision: approved | rejected.
        """
        if decision not in ("approved", "rejected"):
            msg = f"Недопустимое решение: {decision!r}"
            raise ValueError(msg)
        cursor = self._conn.execute(
            "UPDATE lexicon SET status = ?, source = 'review', updated_at = CURRENT_TIMESTAMP WHERE canonical = ?",
            (decision, canonical),
        )
        return cursor.rowcount > 0

    def add_variant(self, canonical: str, variant: str) -> bool:
        """Добавить вариант ASR к термину как решение оператора.

        Запись получает `source='review'`, поэтому повторный `load-project`
        не затрёт добавленный вариант данными листа «Лексикон».

        Returns:
            True, если вариант добавлен; False, если он пуст, совпадает
            с канонической формой или уже есть в списке.
        """
        value = variant.strip()
        if not value or value.casefold() == canonical.casefold():
            return False
        row = self._conn.execute("SELECT variants_json FROM lexicon WHERE canonical = ?", (canonical,)).fetchone()
        existing = [str(item) for item in json.loads(row[0] or "[]")] if row is not None else []
        if any(item.casefold() == value.casefold() for item in existing):
            return False
        existing.append(value)
        payload = json.dumps(existing, ensure_ascii=False)
        if row is None:
            self._conn.execute(
                "INSERT INTO lexicon (canonical, variants_json, definition, status, source)"
                " VALUES (?, ?, NULL, 'approved', 'review')",
                (canonical, payload),
            )
        else:
            self._conn.execute(
                "UPDATE lexicon SET variants_json = ?, source = 'review', updated_at = CURRENT_TIMESTAMP"
                " WHERE canonical = ?",
                (payload, canonical),
            )
        return True


class MarkerRepository:
    """Маркеры неуверенности."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, expression: str, note: str | None, source: str) -> None:
        """Вставить маркер (идемпотентно)."""
        self._conn.execute(
            "INSERT INTO uncertainty_marker (expression, note, source) VALUES (?, ?, ?)"
            " ON CONFLICT(expression) DO UPDATE SET note = excluded.note, source = excluded.source",
            (expression, note, source),
        )

    def count(self) -> int:
        """Число маркеров."""
        row = self._conn.execute("SELECT COUNT(*) FROM uncertainty_marker").fetchone()
        return int(row[0])

    def expressions(self) -> list[str]:
        """Все маркеры неуверенности."""
        rows = self._conn.execute("SELECT expression FROM uncertainty_marker ORDER BY id").fetchall()
        return [str(row[0]) for row in rows]


class GoldenRepository:
    """Золотой набор: реплики L1 и ручной L2."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(
        self,
        interview_code: str | None,
        utterance_ref: str | None,
        text_l1: str,
        text_l2_manual: str | None,
        source: str,
    ) -> None:
        """Вставить или обновить запись золотого набора."""
        self._conn.execute(
            "INSERT INTO golden_sample (interview_code, utterance_ref, text_l1, text_l2_manual, source)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(interview_code, utterance_ref, text_l1) DO UPDATE SET"
            " text_l2_manual = COALESCE(excluded.text_l2_manual, golden_sample.text_l2_manual),"
            " source = excluded.source",
            (interview_code, utterance_ref, text_l1, text_l2_manual, source),
        )

    def count(self) -> int:
        """Число записей."""
        row = self._conn.execute("SELECT COUNT(*) FROM golden_sample").fetchone()
        return int(row[0])


class ReferenceFactRepository:
    """Эталонные факты из листов «Факты_*»."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(
        self,
        workbook_version_id: int,
        sheet_name: str,
        row: dict[str, Any],
    ) -> None:
        """Вставить факт (идемпотентно)."""
        self._conn.execute(
            "INSERT INTO reference_fact"
            " (workbook_version_id, sheet_name, external_id, question, answer, comment, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(workbook_version_id, sheet_name, external_id) DO UPDATE SET"
            " question = excluded.question, answer = excluded.answer,"
            " comment = excluded.comment, status = excluded.status",
            (
                workbook_version_id,
                sheet_name,
                row.get("external_id"),
                row.get("question"),
                row.get("answer"),
                row.get("comment"),
                row.get("status"),
            ),
        )

    def count(self) -> int:
        """Число фактов."""
        row = self._conn.execute("SELECT COUNT(*) FROM reference_fact").fetchone()
        return int(row[0])


class InterviewRepository:
    """Интервью и говорящие."""

    def __init__(self, conn: sqlite3.Connection, groups: GroupsSettings | None = None) -> None:
        self._conn = conn
        self._groups = groups

    def find_by_external_id(self, interview_id: str) -> dict[str, Any] | None:
        """Интервью по внешнему идентификатору."""
        cursor = self._conn.execute("SELECT * FROM interview WHERE id_external = ?", (interview_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return {column[0]: value for column, value in zip(cursor.description or [], row, strict=True)}

    def create(
        self,
        interview_id: str,
        name: str | None,
        respondent_code: str | None,
        group: str | None,
        transcript_source: str | None,
        interview_date: str | None,
    ) -> int:
        """Создать интервью."""
        cursor = self._conn.execute(
            'INSERT INTO interview (id_external, name, respondent_code, "group", transcript_source, interview_date)'
            " VALUES (?, ?, ?, ?, ?, ?)",
            (interview_id, name, respondent_code, _group_value(group, self._groups), transcript_source, interview_date),
        )
        new_id = cursor.lastrowid
        if new_id is None:
            msg = "SQLite не вернул id вставленной строки interview"
            raise sqlite3.Error(msg)
        return int(new_id)

    def update_metadata(
        self,
        interview_db_id: int,
        name: str | None,
        respondent_code: str | None,
        group: str | None,
        transcript_source: str | None,
        interview_date: str | None,
    ) -> None:
        """Обновить метаданные существующего интервью."""
        self._conn.execute(
            'UPDATE interview SET name = ?, respondent_code = ?, "group" = ?, transcript_source = ?,'
            " interview_date = ? WHERE id = ?",
            (
                name,
                respondent_code,
                _group_value(group, self._groups),
                transcript_source,
                interview_date,
                interview_db_id,
            ),
        )

    def add_speaker(self, interview_db_id: int, label: str, role: str, words: int) -> None:
        """Записать говорящего."""
        self._conn.execute(
            "INSERT INTO interview_speaker (interview_id, speaker_label, role, words) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(interview_id, speaker_label) DO UPDATE SET role = excluded.role, words = excluded.words",
            (interview_db_id, label, role, words),
        )

    def speakers(self, interview_id: str) -> list[tuple[str, str, int]]:
        """Роли, сохранённые для интервью: (метка, роль, слов)."""
        rows = self._conn.execute(
            "SELECT s.speaker_label, s.role, s.words FROM interview_speaker s"
            " JOIN interview i ON s.interview_id = i.id WHERE i.id_external = ? ORDER BY s.id",
            (interview_id,),
        ).fetchall()
        return [(str(label), str(role), int(words)) for label, role, words in rows]

    def update_speaker_role(self, interview_db_id: int, label: str, role: str) -> int:
        """Сменить роль говорящего; вернуть число изменённых строк."""
        cursor = self._conn.execute(
            "UPDATE interview_speaker SET role = ? WHERE interview_id = ? AND speaker_label = ?",
            (role, interview_db_id, label),
        )
        return int(cursor.rowcount)

    def turn_line_labels(self, interview_db_id: int) -> list[tuple[int, str | None]]:
        """Номера строк и метки говорящих в записях интервью."""
        rows = self._conn.execute(
            "SELECT line_no, speaker_label FROM turn WHERE interview_id = ? ORDER BY line_no",
            (interview_db_id,),
        ).fetchall()
        return [(int(line_no), None if label is None else str(label)) for line_no, label in rows]

    def fill_turn_labels(self, interview_db_id: int, labels_by_line: dict[int, str]) -> int:
        """Заполнить метки говорящих в записях интервью; вернуть число строк."""
        changed = 0
        for line_no, label in labels_by_line.items():
            cursor = self._conn.execute(
                "UPDATE turn SET speaker_label = ? WHERE interview_id = ? AND line_no = ?",
                (label, interview_db_id, line_no),
            )
            changed += int(cursor.rowcount)
        return changed

    def update_turn_roles(self, interview_db_id: int, label: str, role: str) -> int:
        """Сменить роль записей одного говорящего; вернуть число изменённых строк."""
        cursor = self._conn.execute(
            "UPDATE turn SET speaker_role = ? WHERE interview_id = ? AND speaker_label = ?",
            (role, interview_db_id, label),
        )
        return int(cursor.rowcount)

    def turns_for_utterances(self, interview_db_id: int) -> list[tuple[int, str, str]]:
        """Записи интервью в порядке транскрипта: (id, роль, текст)."""
        rows = self._conn.execute(
            "SELECT id, speaker_role, text FROM turn WHERE interview_id = ? ORDER BY line_no, id",
            (interview_db_id,),
        ).fetchall()
        return [(int(turn_id), str(role), str(text)) for turn_id, role, text in rows]

    def clear_utterances(self, interview_db_id: int) -> int:
        """Удалить реплики интервью перед пересборкой; вернуть число строк."""
        cursor = self._conn.execute("DELETE FROM utterance WHERE interview_id = ?", (interview_db_id,))
        return int(cursor.rowcount)

    def insert_utterance(
        self,
        interview_db_id: int,
        speaker_role: str,
        text: str,
        turn_ids: list[int],
        is_backchannel: bool,
    ) -> int:
        """Записать реплику."""
        cursor = self._conn.execute(
            "INSERT INTO utterance (interview_id, speaker_role, text, turn_ids, is_backchannel) VALUES (?, ?, ?, ?, ?)",
            (interview_db_id, speaker_role, text, json.dumps(turn_ids), int(is_backchannel)),
        )
        new_id = cursor.lastrowid
        if new_id is None:
            msg = "SQLite не вернул id вставленной строки utterance"
            raise sqlite3.Error(msg)
        return int(new_id)

    def new_terms_on_utterances(self, interview_db_id: int) -> int:
        """Сколько кандидатов новых терминов ссылается на реплики интервью."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM new_term n JOIN utterance u ON n.context_utterance_id = u.id"
            " WHERE u.interview_id = ?",
            (interview_db_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def _table_exists(self, name: str) -> bool:
        """Есть ли таблица в схеме (text_unit появляется только в M1b)."""
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def utterance_ids(self, interview_db_id: int) -> list[int]:
        """Идентификаторы реплик интервью."""
        rows = self._conn.execute(
            "SELECT id FROM utterance WHERE interview_id = ? ORDER BY id",
            (interview_db_id,),
        ).fetchall()
        return [int(row[0]) for row in rows]

    def utterances_for_normalize(self, interview_db_id: int) -> list[tuple[int, str, str, int]]:
        """Реплики интервью в порядке транскрипта: (id, роль, текст, is_backchannel)."""
        rows = self._conn.execute(
            "SELECT id, speaker_role, text, is_backchannel FROM utterance WHERE interview_id = ? ORDER BY id",
            (interview_db_id,),
        ).fetchall()
        return [(int(uid), str(role), str(text), int(flag)) for uid, role, text, flag in rows]

    def interview_db_id_by_code(self, interview_code: str) -> int | None:
        """id интервью по внешнему идентификатору (R01)."""
        row = self._conn.execute("SELECT id FROM interview WHERE id_external = ?", (interview_code,)).fetchone()
        return int(row[0]) if row is not None else None

    def group_of(self, interview_db_id: int) -> str | None:
        """Группа респондента интервью (None — не задана)."""
        row = self._conn.execute('SELECT "group" FROM interview WHERE id = ?', (interview_db_id,)).fetchone()
        return None if row is None or row[0] is None else str(row[0])

    def golden_refs_on_utterances(self, interview_db_id: int) -> int:
        """Сколько записей золотого набора ссылается на реплики интервью (utterance:<id>)."""
        rows = self._conn.execute("SELECT utterance_ref FROM golden_sample WHERE utterance_ref IS NOT NULL").fetchall()
        ids = set(self.utterance_ids(interview_db_id))
        linked = 0
        for (ref,) in rows:
            prefix, _, value = str(ref).partition(":")
            if prefix != "utterance" or not value.isdigit():
                continue
            if int(value) in ids:
                linked += 1
        return linked

    def text_unit_refs_on_utterances(self, interview_db_id: int) -> int:
        """Сколько единиц L2 ссылается на реплики интервью (таблица появляется в M1b)."""
        if not self._table_exists("text_unit"):
            return 0
        row = self._conn.execute(
            "SELECT COUNT(*) FROM text_unit t JOIN utterance u ON t.utterance_id = u.id WHERE u.interview_id = ?",
            (interview_db_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def annotation_refs_on_utterances(self, interview_db_id: int) -> int:
        """Сколько аннотаций ссылается на фрагменты реплик интервью.

        Фрагмент (M1c) привязан к реплике `utterance`, а не к записи `turn`:
        после миграции 0006 колонки `fragment.turn_id` больше нет.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) FROM annotation a"
            " JOIN fragment f ON a.fragment_id = f.id"
            " JOIN utterance u ON f.utterance_id = u.id"
            " WHERE u.interview_id = ?",
            (interview_db_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def speaker_role_snapshot(self, interview_db_id: int) -> dict[str, str]:
        """Сохранённые роли: карта «метка говорящего -> роль»."""
        rows = self._conn.execute(
            "SELECT speaker_label, role FROM interview_speaker WHERE interview_id = ?",
            (interview_db_id,),
        ).fetchall()
        return {str(label): str(role) for label, role in rows}

    def turn_roles_by_label(self, interview_db_id: int) -> dict[str, set[str]]:
        """Роли записей по меткам: карта «метка -> множество ролей»."""
        rows = self._conn.execute(
            "SELECT speaker_label, speaker_role FROM turn WHERE interview_id = ? AND speaker_label IS NOT NULL",
            (interview_db_id,),
        ).fetchall()
        result: dict[str, set[str]] = {}
        for label, role in rows:
            result.setdefault(str(label), set()).add(str(role))
        return result

    def count_turns_without_label(self, interview_db_id: int) -> int:
        """Сколько записей интервью осталось без метки говорящего."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM turn WHERE interview_id = ? AND speaker_label IS NULL",
            (interview_db_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0


class NewTermRepository:
    """Кандидаты новых терминов для review-terms (M1b)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, surface_form: str, interview_id: int | None, context_utterance_id: int | None) -> int:
        """Добавить кандидата."""
        cursor = self._conn.execute(
            "INSERT INTO new_term (interview_id, surface_form, context_utterance_id) VALUES (?, ?, ?)"
            " ON CONFLICT(surface_form, context_utterance_id) DO NOTHING",
            (interview_id, surface_form, context_utterance_id),
        )
        if cursor.rowcount == 0:
            row = self._conn.execute(
                "SELECT id FROM new_term WHERE surface_form = ? AND context_utterance_id IS ?",
                (surface_form, context_utterance_id),
            ).fetchone()
            return int(row[0])
        new_id = cursor.lastrowid
        if new_id is None:
            msg = "SQLite не вернул id вставленной строки new_term"
            raise sqlite3.Error(msg)
        return int(new_id)

    def proposed(self) -> list[tuple[int, str, str | None]]:
        """Ожидающие решения кандидаты: (id, термин, контекст)."""
        rows = self._conn.execute(
            "SELECT id, surface_form, context_utterance_id FROM new_term WHERE status = 'proposed' ORDER BY id"
        ).fetchall()
        return [(int(i), str(form), None if ctx is None else str(ctx)) for i, form, ctx in rows]

    def proposed_on_current_l2(self) -> list[tuple[int, str, str | None]]:
        """Кандидаты, форма которых есть в актуальной версии L2 своего интервью.

        Актуальная версия локальна для интервью: у R01 это последняя записанная
        версия R01, у другого интервью — своя. Сравнивать все кандидаты с одной
        глобальной версией нельзя: тогда кандидаты интервью, обработанного более
        новым пайплайном (одна версия), выпали бы из очереди.

        Форма ищется по всему актуальному тексту интервью, а не только по
        реплике-контексту: термин мог переехать в соседнюю реплику при
        исправлении ASR, и терять его из очереди нельзя.

        Сравнение идёт в Python через `casefold`: `lower()` в SQLite понимает
        только ASCII, поэтому кириллица в SQL не понижается.

        Кандидат без реплики-контекста остаётся в очереди: проверить нечем.
        """
        rows = self._conn.execute(
            """
            SELECT n.id, n.surface_form, n.context_utterance_id, u.interview_id
            FROM new_term n
            LEFT JOIN utterance u ON u.id = n.context_utterance_id
            WHERE n.status = 'proposed'
            ORDER BY n.id
            """
        ).fetchall()

        interview_texts: dict[int, str] = {}
        for interview_id in {int(row[3]) for row in rows if row[3] is not None}:
            units = self._conn.execute(
                """
                SELECT t.text FROM text_unit t JOIN utterance u ON u.id = t.utterance_id
                WHERE u.interview_id = ? AND t.version = (
                    SELECT MAX(t2.version) FROM text_unit t2
                    JOIN utterance u2 ON u2.id = t2.utterance_id
                    WHERE u2.interview_id = u.interview_id
                )
                """,
                (interview_id,),
            ).fetchall()
            interview_texts[interview_id] = " ".join(str(unit[0]) for unit in units).casefold()

        result: list[tuple[int, str, str | None]] = []
        for term_id, form, context, interview_id in rows:
            if context is None or interview_id is None:
                result.append((int(term_id), str(form), None if context is None else str(context)))
                continue
            if str(form).casefold() in interview_texts.get(int(interview_id), ""):
                result.append((int(term_id), str(form), str(context)))
        return result

    def decide(self, term_id: int, decision: str) -> None:
        """Зафиксировать решение по кандидату."""
        if decision not in ("approved", "rejected"):
            msg = f"Недопустимое решение: {decision!r}"
            raise ValueError(msg)
        self._conn.execute("UPDATE new_term SET status = ? WHERE id = ?", (decision, term_id))


@dataclass(frozen=True)
class UtteranceRow:
    """Реплика L1 для нормализации."""

    id: int
    speaker_role: str
    text: str
    is_backchannel: bool


class TextUnitRepository:
    """Слой L2: нормализованные единицы по репликам."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def max_version(self, utterance_id: int) -> int:
        """Наибольшая существующая версия L2 для реплики (0 — записей нет)."""
        row = self._conn.execute(
            "SELECT MAX(version) FROM text_unit WHERE utterance_id = ?",
            (utterance_id,),
        ).fetchone()
        return int(row[0]) if row is not None and row[0] is not None else 0

    def exists(self, utterance_id: int, version: int) -> bool:
        """Есть ли уже единица заданной версии."""
        row = self._conn.execute(
            "SELECT 1 FROM text_unit WHERE utterance_id = ? AND version = ?",
            (utterance_id, version),
        ).fetchone()
        return row is not None

    def insert(
        self,
        utterance_id: int,
        version: int,
        text: str,
        status: str,
        asr_quality: str,
        changes_json: str,
        uncertain_spans_json: str,
        fillers_removed: int,
        stutter_removed: int,
        repeat_unclear: int,
        run_id: int | None,
    ) -> int:
        """Записать единицу L2."""
        cursor = self._conn.execute(
            "INSERT INTO text_unit (utterance_id, version, text, status, asr_quality, changes_json,"
            " uncertain_spans_json, fillers_removed, stutter_removed, repeat_unclear, run_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                utterance_id,
                version,
                text,
                status,
                asr_quality,
                changes_json,
                uncertain_spans_json,
                fillers_removed,
                stutter_removed,
                repeat_unclear,
                run_id,
            ),
        )
        new_id = cursor.lastrowid
        if new_id is None:
            msg = "SQLite не вернул id вставленной строки text_unit"
            raise sqlite3.Error(msg)
        return int(new_id)

    def counts_by_asr_quality(self, interview_db_id: int) -> dict[str, int]:
        """Распределение asr_quality по единицам интервью."""
        rows = self._conn.execute(
            "SELECT t.asr_quality, COUNT(*) FROM text_unit t JOIN utterance u ON t.utterance_id = u.id"
            " WHERE u.interview_id = ? GROUP BY t.asr_quality",
            (interview_db_id,),
        ).fetchall()
        return {str(quality): int(count) for quality, count in rows}

    def max_version_for_interview(self, interview_db_id: int) -> int:
        """Наибольшая записанная версия L2 для интервью (0 — записей нет)."""
        row = self._conn.execute(
            "SELECT MAX(t.version) FROM text_unit t JOIN utterance u ON t.utterance_id = u.id WHERE u.interview_id = ?",
            (interview_db_id,),
        ).fetchone()
        return int(row[0]) if row is not None and row[0] is not None else 0

    def count_for_interview(self, interview_db_id: int, version: int = 1) -> int:
        """Сколько единиц L2 записано для интервью."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM text_unit t JOIN utterance u ON t.utterance_id = u.id"
            " WHERE u.interview_id = ? AND t.version = ?",
            (interview_db_id, version),
        ).fetchone()
        return int(row[0]) if row is not None else 0


class RunRepository:
    """Запись прогонов пайплайна."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def start(
        self,
        model: str,
        prompt_hash: str,
        codebook_version_id: int | None,
        code_version: str,
        params_json: str,
    ) -> int:
        """Записать начало прогона и вернуть его id."""
        cursor = self._conn.execute(
            "INSERT INTO run (model, prompt_hash, codebook_version_id, code_version, params_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (model, prompt_hash, codebook_version_id, code_version, params_json),
        )
        run_id = cursor.lastrowid
        if run_id is None:
            msg = "SQLite не вернул id вставленной строки run"
            raise sqlite3.Error(msg)
        return int(run_id)

    def latest_codebook_version_id(self) -> int | None:
        """Последняя версия кодбука (для привязки прогона)."""
        row = self._conn.execute("SELECT id FROM codebook_version ORDER BY id DESC LIMIT 1").fetchone()
        return int(row[0]) if row is not None else None


@dataclass(frozen=True)
class GuideQuestion:
    """Вопрос гайда, пригодный для привязки реплик."""

    id: int
    identifier: str
    block: str | None
    group: str | None
    wording: str


@dataclass(frozen=True)
class UtteranceWithL2:
    """Реплика интервью вместе с её L2 нужной версии."""

    id: int
    speaker_role: str
    text: str
    is_backchannel: bool
    text_unit_id: int | None
    l2_text: str | None
    asr_quality: str | None


@dataclass(frozen=True)
class FragmentRow:
    """Фрагмент L2 для записи в БД."""

    utterance_id: int
    text_unit_id: int
    start: int
    end: int
    text: str
    question_id: int | None
    prior_interviewer_turn_id: int | None
    asr_quality: str


class StructureRepository:
    """Привязка реплик к вопросам гайда и фрагменты L2 (M1c)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def guide_questions(self, codebook_version_id: int | None = None) -> list[GuideQuestion]:
        """Вопросы гайда с непустой формулировкой.

        Args:
            codebook_version_id: версия кодбука; None — последняя загруженная.
        """
        version_id = codebook_version_id if codebook_version_id is not None else self.latest_codebook_version_id()
        if version_id is None:
            return []
        rows = self._conn.execute(
            'SELECT id, identifier, block, "group", wording FROM question'
            " WHERE codebook_version_id = ? AND wording IS NOT NULL AND TRIM(wording) <> ''"
            " ORDER BY id",
            (version_id,),
        ).fetchall()
        return [
            GuideQuestion(
                id=int(qid),
                identifier=str(identifier),
                block=None if block is None else str(block),
                group=None if group is None else str(group),
                wording=str(wording),
            )
            for qid, identifier, block, group, wording in rows
        ]

    def latest_codebook_version_id(self) -> int | None:
        """Последняя версия кодбука."""
        row = self._conn.execute("SELECT id FROM codebook_version ORDER BY id DESC LIMIT 1").fetchone()
        return int(row[0]) if row is not None else None

    def utterances_with_l2(self, interview_db_id: int, version: int) -> list[UtteranceWithL2]:
        """Реплики интервью в порядке транскрипта вместе с L2 заданной версии.

        Реплики без L2 тоже возвращаются (`text_unit_id = None`): без L2
        фрагментов нет, но привязка к вопросу у реплики интервьюера есть.
        """
        rows = self._conn.execute(
            "SELECT u.id, u.speaker_role, u.text, u.is_backchannel, t.id, t.text, t.asr_quality"
            " FROM utterance u"
            " LEFT JOIN text_unit t ON t.utterance_id = u.id AND t.version = ?"
            " WHERE u.interview_id = ?"
            " ORDER BY u.id",
            (version, interview_db_id),
        ).fetchall()
        return [
            UtteranceWithL2(
                id=int(uid),
                speaker_role=str(role),
                text=str(text),
                is_backchannel=bool(flag),
                text_unit_id=None if tu_id is None else int(tu_id),
                l2_text=None if tu_text is None else str(tu_text),
                asr_quality=None if quality is None else str(quality),
            )
            for uid, role, text, flag, tu_id, tu_text, quality in rows
        ]

    def last_turn_id(self, utterance_id: int) -> int | None:
        """id последней записи транскрипта в реплике (для prior_interviewer_turn_id)."""
        row = self._conn.execute("SELECT turn_ids FROM utterance WHERE id = ?", (utterance_id,)).fetchone()
        if row is None:
            return None
        try:
            values = json.loads(str(row[0]))
        except json.JSONDecodeError:
            return None
        if not isinstance(values, list) or not values:
            return None
        last = values[-1]
        return int(last) if isinstance(last, int) else None

    def set_utterance_binding(
        self,
        utterance_id: int,
        question_id: int | None,
        is_probe: bool,
        prior_interviewer_turn_id: int | None,
    ) -> None:
        """Записать привязку реплики к вопросу гайда."""
        self._conn.execute(
            "UPDATE utterance SET question_id = ?, is_probe = ?, prior_interviewer_turn_id = ? WHERE id = ?",
            (question_id, int(is_probe), prior_interviewer_turn_id, utterance_id),
        )

    def clear_fragments(self, interview_db_id: int) -> int:
        """Удалить фрагменты интервью перед пересборкой; вернуть число строк."""
        cursor = self._conn.execute(
            "DELETE FROM fragment WHERE utterance_id IN (SELECT id FROM utterance WHERE interview_id = ?)",
            (interview_db_id,),
        )
        return int(cursor.rowcount)

    def insert_fragment(self, row: FragmentRow) -> int:
        """Записать фрагмент L2."""
        cursor = self._conn.execute(
            'INSERT INTO fragment (utterance_id, text_unit_id, "start", "end", text,'
            " question_id, prior_interviewer_turn_id, asr_quality)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.utterance_id,
                row.text_unit_id,
                row.start,
                row.end,
                row.text,
                row.question_id,
                row.prior_interviewer_turn_id,
                row.asr_quality,
            ),
        )
        new_id = cursor.lastrowid
        if new_id is None:
            msg = "SQLite не вернул id вставленной строки fragment"
            raise sqlite3.Error(msg)
        return int(new_id)

    def fragments_of_interview(self, interview_db_id: int) -> list[tuple[int, int, int, str, int | None]]:
        """Фрагменты интервью: (id, utterance_id, text_unit_id, text, question_id)."""
        rows = self._conn.execute(
            "SELECT f.id, f.utterance_id, f.text_unit_id, f.text, f.question_id"
            " FROM fragment f JOIN utterance u ON f.utterance_id = u.id"
            " WHERE u.interview_id = ? ORDER BY f.utterance_id, f.start",
            (interview_db_id,),
        ).fetchall()
        return [
            (int(fid), int(uid), int(tuid), str(text), None if qid is None else int(qid))
            for fid, uid, tuid, text, qid in rows
        ]

    def fragment_l2_versions(self, interview_db_id: int) -> set[int]:
        """Версии L2, на которые ссылаются записанные фрагменты интервью."""
        rows = self._conn.execute(
            "SELECT DISTINCT t.version FROM fragment f"
            " JOIN utterance u ON f.utterance_id = u.id"
            " JOIN text_unit t ON f.text_unit_id = t.id"
            " WHERE u.interview_id = ?",
            (interview_db_id,),
        ).fetchall()
        return {int(row[0]) for row in rows}

    def utterances_per_question(self, interview_db_id: int) -> dict[str | None, int]:
        """Сколько реплик респондента привязано к каждому вопросу (None — вне гайда)."""
        rows = self._conn.execute(
            "SELECT q.identifier, COUNT(*)"
            " FROM utterance u LEFT JOIN question q ON u.question_id = q.id"
            " WHERE u.interview_id = ? AND u.speaker_role = 'RESPONDENT'"
            " GROUP BY q.identifier",
            (interview_db_id,),
        ).fetchall()
        return {None if identifier is None else str(identifier): int(count) for identifier, count in rows}
