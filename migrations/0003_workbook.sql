-- Миграция 0003 (рабочая книга): версии книги, листы, вопросы с текстом,
-- связи вопрос-код, словарь замен, лексикон, маркеры, золотой набор, факты.
-- FK проверяются в конце через PRAGMA foreign_key_check.

PRAGMA foreign_keys=OFF;

-- Версия прочитанной рабочей книги: хеш файла и хеш кодбуковых листов.
CREATE TABLE IF NOT EXISTS workbook_version (
    id INTEGER PRIMARY KEY,
    workbook_path TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    codebook_hash TEXT NOT NULL,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Хеши отдельных листов в составе версии книги.
CREATE TABLE IF NOT EXISTS workbook_sheet (
    id INTEGER PRIMARY KEY,
    workbook_version_id INTEGER NOT NULL REFERENCES workbook_version(id),
    sheet_name TEXT NOT NULL,
    header_hash TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    n_rows INTEGER NOT NULL DEFAULT 0,
    UNIQUE(workbook_version_id, sheet_name)
);

-- Кодбук привязан к версии книги и имеет собственный хеш (только кодбуковые листы).
ALTER TABLE codebook_version ADD COLUMN workbook_version_id INTEGER REFERENCES workbook_version(id);
ALTER TABLE codebook_version ADD COLUMN codebook_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS ux_codebook_version_hash ON codebook_version(codebook_hash);

-- Вопросы гайда: текст и служебные поля из листа «Вопросы_и_коды».
ALTER TABLE question ADD COLUMN block TEXT;
ALTER TABLE question ADD COLUMN "group" TEXT;
ALTER TABLE question ADD COLUMN wording TEXT;
ALTER TABLE question ADD COLUMN goal TEXT;
ALTER TABLE question ADD COLUMN notes TEXT;

-- Коды: поля листа «Коды».
ALTER TABLE code ADD COLUMN kind TEXT;
ALTER TABLE code ADD COLUMN title TEXT;
ALTER TABLE code ADD COLUMN tiebreak_rule TEXT;

-- Связь «вопрос -> код» из листов «Вопросы_и_коды» и «Коды».
CREATE TABLE IF NOT EXISTS question_code (
    id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL REFERENCES question(id),
    code_id INTEGER NOT NULL REFERENCES code(id),
    UNIQUE(question_id, code_id)
);

-- Справочник атрибутов разметки из листа «Атрибуты».
CREATE TABLE IF NOT EXISTS attribute_definition (
    id INTEGER PRIMARY KEY,
    codebook_version_id INTEGER NOT NULL REFERENCES codebook_version(id),
    name TEXT NOT NULL,
    kind TEXT,
    description TEXT,
    UNIQUE(codebook_version_id, name)
);

-- Словарь замен для интервьюера (лист «Словарь_замен»).
CREATE TABLE IF NOT EXISTS glossary_term (
    id INTEGER PRIMARY KEY,
    codebook_version_id INTEGER NOT NULL REFERENCES codebook_version(id),
    term TEXT NOT NULL,
    replacement TEXT,
    UNIQUE(codebook_version_id, term)
);

-- Лексикон терминов (лист «Лексикон» + решения review-terms).
-- source: workbook | review — решения review-terms имеют приоритет.
CREATE TABLE IF NOT EXISTS lexicon (
    id INTEGER PRIMARY KEY,
    canonical TEXT NOT NULL,
    variants_json TEXT NOT NULL DEFAULT '[]',
    definition TEXT,
    status TEXT NOT NULL DEFAULT 'proposed',
    source TEXT NOT NULL DEFAULT 'workbook' CHECK (source IN ('workbook', 'review')),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(canonical)
);

-- Маркеры неуверенности (лист «Маркеры_неуверенности» или умолчания).
CREATE TABLE IF NOT EXISTS uncertainty_marker (
    id INTEGER PRIMARY KEY,
    expression TEXT NOT NULL UNIQUE,
    note TEXT,
    source TEXT NOT NULL DEFAULT 'workbook' CHECK (source IN ('workbook', 'default'))
);

-- Золотой набор: реплики L1 и ручной L2 (лист «Золотой_набор» или xlsx экспорта).
CREATE TABLE IF NOT EXISTS golden_sample (
    id INTEGER PRIMARY KEY,
    interview_code TEXT,
    utterance_ref TEXT,
    text_l1 TEXT NOT NULL,
    text_l2_manual TEXT,
    source TEXT NOT NULL DEFAULT 'workbook',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(interview_code, utterance_ref, text_l1)
);

-- Эталонные факты из листов с префиксом «Факты_».
CREATE TABLE IF NOT EXISTS reference_fact (
    id INTEGER PRIMARY KEY,
    workbook_version_id INTEGER NOT NULL REFERENCES workbook_version(id),
    sheet_name TEXT NOT NULL,
    external_id TEXT NOT NULL,
    question TEXT,
    answer TEXT,
    comment TEXT,
    status TEXT,
    UNIQUE(workbook_version_id, sheet_name, external_id)
);

-- Говорящие интервью: метки, роли и число слов (init-interview).
CREATE TABLE IF NOT EXISTS interview_speaker (
    id INTEGER PRIMARY KEY,
    interview_id INTEGER NOT NULL REFERENCES interview(id),
    speaker_label TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('INTERVIEWER', 'RESPONDENT', 'OBSERVER')),
    words INTEGER NOT NULL DEFAULT 0,
    UNIQUE(interview_id, speaker_label)
);

-- Дата интервью из заголовка транскрипта или листа «Интервью».
ALTER TABLE interview ADD COLUMN interview_date TEXT;

-- Кандидаты новых терминов (review-terms; используется в M1b).
CREATE TABLE IF NOT EXISTS new_term (
    id INTEGER PRIMARY KEY,
    interview_id INTEGER REFERENCES interview(id),
    surface_form TEXT NOT NULL,
    context_utterance_id INTEGER REFERENCES utterance(id),
    status TEXT NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed', 'approved', 'rejected')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(surface_form, context_utterance_id)
);

-- Проверка целостности внешних ключей: при нарушениях CHECK не пропустит.
DROP TABLE IF EXISTS _guard_fk;
CREATE TABLE _guard_fk (n INTEGER NOT NULL CHECK (n = 0));
INSERT INTO _guard_fk SELECT COUNT(*) FROM pragma_foreign_key_check;
DROP TABLE _guard_fk;

PRAGMA foreign_keys=ON;
