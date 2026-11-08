-- Миграция 0006 (M1c): привязка реплик к вопросам гайда и фрагменты по L2.
-- fragment пересоздаётся по схеме SPEC-M1 раздел M1c: единица анализа — реплика
-- (utterance) и её слой L2 (text_unit), а не запись транскрипта (turn).
-- Гарды пустоты: пересоздание допустимо, только если на fragment никто не ссылается.

PRAGMA foreign_keys=OFF;

-- Гард пустоты: CHECK не даст вставить count > 0.
DROP TABLE IF EXISTS _guard_empty;
CREATE TABLE _guard_empty (n INTEGER NOT NULL CHECK (n = 0));
INSERT INTO _guard_empty SELECT COUNT(*) FROM fragment;
INSERT INTO _guard_empty SELECT COUNT(*) FROM annotation;
INSERT INTO _guard_empty SELECT COUNT(*) FROM embedding;
DROP TABLE _guard_empty;

DROP TABLE IF EXISTS fragment;

-- Фрагмент смысловой единицы: дословный срез текста L2.
-- start/end — смещения в text_unit.text (в символах, Python-срез), end не включён.
-- question_id и prior_interviewer_turn_id дублируются из реплики: фрагмент
-- кодируется отдельно, и его контекст должен читаться без join по utterance.
CREATE TABLE fragment (
    id INTEGER PRIMARY KEY,
    utterance_id INTEGER NOT NULL REFERENCES utterance(id),
    text_unit_id INTEGER NOT NULL REFERENCES text_unit(id),
    "start" INTEGER NOT NULL,
    "end" INTEGER NOT NULL,
    text TEXT NOT NULL,
    question_id INTEGER REFERENCES question(id),
    prior_interviewer_turn_id INTEGER REFERENCES turn(id),
    asr_quality TEXT NOT NULL DEFAULT 'ok' CHECK (asr_quality IN ('ok', 'doubtful', 'garbled')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK ("start" >= 0 AND "end" >= "start")
);

CREATE INDEX IF NOT EXISTS idx_fragment_utterance ON fragment(utterance_id);
CREATE INDEX IF NOT EXISTS idx_fragment_text_unit ON fragment(text_unit_id);
CREATE INDEX IF NOT EXISTS idx_fragment_question ON fragment(question_id);

-- Привязка реплик к вопросам гайда (M1c).
-- question_id = NULL: реплика вне гайда, а не «ближайший вопрос».
-- is_probe = 1: уточнение интервьюера к основному вопросу.
-- prior_interviewer_turn_id: последняя реплика интервьюера перед репликой
-- респондента; нужна для признака «спонтанно / по подсказке» в M2.
ALTER TABLE utterance ADD COLUMN question_id INTEGER REFERENCES question(id);
ALTER TABLE utterance ADD COLUMN is_probe INTEGER NOT NULL DEFAULT 0;
ALTER TABLE utterance ADD COLUMN prior_interviewer_turn_id INTEGER REFERENCES turn(id);

CREATE INDEX IF NOT EXISTS idx_utterance_question ON utterance(question_id);

-- Проверка целостности внешних ключей: при нарушениях CHECK не пропустит.
DROP TABLE IF EXISTS _guard_fk;
CREATE TABLE _guard_fk (n INTEGER NOT NULL CHECK (n = 0));
INSERT INTO _guard_fk SELECT COUNT(*) FROM pragma_foreign_key_check;
DROP TABLE _guard_fk;

PRAGMA foreign_keys=ON;
