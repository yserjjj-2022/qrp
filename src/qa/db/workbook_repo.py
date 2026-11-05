"""Репозитории рабочей книги: версии, кодбук, темы, лексикон, факты, интервью.

Весь SQL по этим сущностям живёт здесь; пайплайн и CLI его не содержат.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from qa.pipeline.workbook import WorkbookData


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

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

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
                row.get("group"),
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

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

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
            (interview_id, name, respondent_code, group, transcript_source, interview_date),
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
            (name, respondent_code, group, transcript_source, interview_date, interview_db_id),
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

    def decide(self, term_id: int, decision: str) -> None:
        """Зафиксировать решение по кандидату."""
        if decision not in ("approved", "rejected"):
            msg = f"Недопустимое решение: {decision!r}"
            raise ValueError(msg)
        self._conn.execute("UPDATE new_term SET status = ? WHERE id = ?", (decision, term_id))
