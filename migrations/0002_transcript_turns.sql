-- Миграция 0002 (M1a): перестройка turn, таблица utterance, колонки interview.
-- Гарды: таблицы turn, fragment, annotation должны быть пусты, иначе миграция останавливается.
-- FK отключаются на время пересоздания, в конце — PRAGMA foreign_key_check.

PRAGMA foreign_keys=OFF;

-- Гард пустоты: CHECK не даст вставить count > 0
DROP TABLE IF EXISTS _guard_empty;
CREATE TABLE _guard_empty (n INTEGER NOT NULL CHECK (n = 0));
INSERT INTO _guard_empty SELECT COUNT(*) FROM turn;
INSERT INTO _guard_empty SELECT COUNT(*) FROM fragment;
INSERT INTO _guard_empty SELECT COUNT(*) FROM annotation;
DROP TABLE _guard_empty;

-- Перестройка turn: новый формат записи транскрипта
DROP TABLE IF EXISTS fragment;
DROP TABLE IF EXISTS turn;

CREATE TABLE turn (
    id INTEGER PRIMARY KEY,
    interview_id INTEGER NOT NULL REFERENCES interview(id),
    line_no INTEGER NOT NULL,
    start_sec INTEGER NOT NULL,
    speaker_role TEXT NOT NULL CHECK (speaker_role IN ('INTERVIEWER', 'RESPONDENT', 'OBSERVER')),
    text TEXT NOT NULL,
    pii_flag INTEGER NOT NULL DEFAULT 0
);

-- Сборные реплики: объединение подряд идущих записей одного говорящего
CREATE TABLE utterance (
    id INTEGER PRIMARY KEY,
    interview_id INTEGER NOT NULL REFERENCES interview(id),
    speaker_role TEXT NOT NULL CHECK (speaker_role IN ('INTERVIEWER', 'RESPONDENT', 'OBSERVER')),
    text TEXT NOT NULL,
    turn_ids TEXT NOT NULL,
    is_backchannel INTEGER NOT NULL DEFAULT 0
);

-- fragment пересоздаётся в прежнем виде (ссылается на turn); перестройка — в 0004 (M1c)
CREATE TABLE fragment (
    id INTEGER PRIMARY KEY,
    turn_id INTEGER NOT NULL REFERENCES turn(id),
    text TEXT NOT NULL,
    start_pos INTEGER,
    end_pos INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Метаданные интервью (M1a)
ALTER TABLE interview ADD COLUMN respondent_code TEXT;
ALTER TABLE interview ADD COLUMN "group" TEXT;
ALTER TABLE interview ADD COLUMN transcript_source TEXT;

-- Проверка целостности внешних ключей: при нарушениях CHECK не пропустит
DROP TABLE IF EXISTS _guard_fk;
CREATE TABLE _guard_fk (n INTEGER NOT NULL CHECK (n = 0));
INSERT INTO _guard_fk SELECT COUNT(*) FROM pragma_foreign_key_check;
DROP TABLE _guard_fk;

PRAGMA foreign_keys=ON;
