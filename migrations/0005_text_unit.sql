-- Миграция 0005 (M1b): слой L2, одна нормализованная единица на реплику.
-- Связь L1 -> L2 один к одному: UNIQUE(utterance_id, version).
-- status = proposed: проход модели принят; rejected: вывод отклонён автопроверкой,
-- в text лежит результат детерминированного прохода (он и есть L2).
-- review_status и reviewed_at пока не используются: ручное ревью отложено.

CREATE TABLE IF NOT EXISTS text_unit (
    id INTEGER PRIMARY KEY,
    utterance_id INTEGER NOT NULL REFERENCES utterance(id),
    version INTEGER NOT NULL DEFAULT 1,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed', 'rejected')),
    asr_quality TEXT NOT NULL DEFAULT 'ok' CHECK (asr_quality IN ('ok', 'doubtful', 'garbled')),
    changes_json TEXT NOT NULL DEFAULT '[]',
    uncertain_spans_json TEXT NOT NULL DEFAULT '[]',
    fillers_removed INTEGER NOT NULL DEFAULT 0,
    stutter_removed INTEGER NOT NULL DEFAULT 0,
    repeat_unclear INTEGER NOT NULL DEFAULT 0,
    run_id INTEGER REFERENCES run(id),
    review_status TEXT,
    reviewed_at TIMESTAMP,
    UNIQUE(utterance_id, version)
);

CREATE INDEX IF NOT EXISTS idx_text_unit_utterance ON text_unit(utterance_id);

-- Проверка целостности внешних ключей: при нарушениях CHECK не пропустит.
DROP TABLE IF EXISTS _guard_fk;
CREATE TABLE _guard_fk (n INTEGER NOT NULL CHECK (n = 0));
INSERT INTO _guard_fk SELECT COUNT(*) FROM pragma_foreign_key_check;
DROP TABLE _guard_fk;
