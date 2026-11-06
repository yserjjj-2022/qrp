-- Миграция 0004 (M1a): метка говорящего в записи транскрипта.
-- Нужна, чтобы правка роли в листе «Роли_говорящих» доезжала до turn и utterance:
-- без метки нельзя понять, какие записи относятся к какому говорящему.
-- Колонка nullable: существующие записи остаются валидными и заполняются backfill-ом.

ALTER TABLE turn ADD COLUMN speaker_label TEXT;

CREATE INDEX IF NOT EXISTS idx_turn_speaker_label ON turn(interview_id, speaker_label);

-- Проверка целостности внешних ключей: при нарушениях CHECK не пропустит.
DROP TABLE IF EXISTS _guard_fk;
CREATE TABLE _guard_fk (n INTEGER NOT NULL CHECK (n = 0));
INSERT INTO _guard_fk SELECT COUNT(*) FROM pragma_foreign_key_check;
DROP TABLE _guard_fk;
