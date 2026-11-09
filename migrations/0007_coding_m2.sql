-- Миграция 0007 (M2): кодирование фрагментов кодами кодбука.
-- Дропов нет: CREATE TABLE и ALTER TABLE ADD COLUMN, поэтому подтверждение
-- деструктивной операции не требуется (CONSTITUTION §4).
-- annotation сейчас пуста, новые колонки добавляются без переноса данных.

-- Итог кодирования фрагмента в прогоне. Нужен, чтобы отличить «код не
-- поставлен» от «фрагмент пропущен как искажённый» и от «модель не ответила»:
-- без этой таблицы три случая в отчёте неразличимы.
CREATE TABLE IF NOT EXISTS fragment_coding (
    id INTEGER PRIMARY KEY,
    fragment_id INTEGER NOT NULL REFERENCES fragment(id),
    run_id INTEGER NOT NULL REFERENCES run(id),
    status TEXT NOT NULL CHECK (status IN ('coded', 'skipped_garbled', 'unmarked', 'failed')),
    note TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(fragment_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_fragment_coding_fragment ON fragment_coding(fragment_id);
CREATE INDEX IF NOT EXISTS idx_fragment_coding_run ON fragment_coding(run_id);

-- Связь аннотации с итогом прогона и атрибуты разметки.
-- confidence остаётся неиспользованной: её смысл заменяет явный reliability.
ALTER TABLE annotation ADD COLUMN coding_id INTEGER REFERENCES fragment_coding(id);
ALTER TABLE annotation ADD COLUMN interpretation TEXT;
ALTER TABLE annotation ADD COLUMN interpretation_only INTEGER NOT NULL DEFAULT 0
    CHECK (interpretation_only IN (0, 1));
ALTER TABLE annotation ADD COLUMN order_kind TEXT CHECK (order_kind IN ('spontaneous', 'prompted'));
ALTER TABLE annotation ADD COLUMN scenario TEXT CHECK (scenario IN ('real', 'imaginary'));
ALTER TABLE annotation ADD COLUMN reliability INTEGER CHECK (reliability BETWEEN 1 AND 3);
ALTER TABLE annotation ADD COLUMN reliability_note TEXT;
ALTER TABLE annotation ADD COLUMN oth_description TEXT;

CREATE INDEX IF NOT EXISTS idx_annotation_coding ON annotation(coding_id);

-- Проверка целостности внешних ключей: при нарушениях CHECK не пропустит.
DROP TABLE IF EXISTS _guard_fk;
CREATE TABLE _guard_fk (n INTEGER NOT NULL CHECK (n = 0));
INSERT INTO _guard_fk SELECT COUNT(*) FROM pragma_foreign_key_check;
DROP TABLE _guard_fk;
