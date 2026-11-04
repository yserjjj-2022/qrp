-- Миграция 0001: создать все таблицы каркаса проекта
-- Всего 15 таблиц

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- 1. interview
CREATE TABLE IF NOT EXISTS interview (
    id INTEGER PRIMARY KEY,
    id_external TEXT NOT NULL UNIQUE,
    name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. turn
CREATE TABLE IF NOT EXISTS turn (
    id INTEGER PRIMARY KEY,
    interview_id INTEGER NOT NULL REFERENCES interview(id),
    speaker TEXT,
    text TEXT NOT NULL,
    question_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. fragment
CREATE TABLE IF NOT EXISTS fragment (
    id INTEGER PRIMARY KEY,
    turn_id INTEGER NOT NULL REFERENCES turn(id),
    text TEXT NOT NULL,
    start_pos INTEGER,
    end_pos INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. codebook_version
CREATE TABLE IF NOT EXISTS codebook_version (
    id INTEGER PRIMARY KEY,
    file_hash TEXT NOT NULL,
    extensions_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. code
CREATE TABLE IF NOT EXISTS code (
    id INTEGER PRIMARY KEY,
    codebook_version_id INTEGER NOT NULL REFERENCES codebook_version(id),
    family TEXT,
    name TEXT NOT NULL,
    definition TEXT,
    include TEXT,
    exclude TEXT,
    tiebreak INTEGER,
    applies_to TEXT,
    origin TEXT NOT NULL DEFAULT 'base' CHECK (origin IN ('base', 'inductive')),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'merged', 'retired')),
    supersedes_id INTEGER REFERENCES code(id),
    UNIQUE(codebook_version_id, name)
);

-- 6. attribute
CREATE TABLE IF NOT EXISTS attribute (
    id INTEGER PRIMARY KEY,
    code_id INTEGER NOT NULL REFERENCES code(id),
    name TEXT NOT NULL,
    value TEXT,
    UNIQUE(code_id, name)
);

-- 7. question
CREATE TABLE IF NOT EXISTS question (
    id INTEGER PRIMARY KEY,
    codebook_version_id INTEGER NOT NULL REFERENCES codebook_version(id),
    identifier TEXT NOT NULL,
    UNIQUE(codebook_version_id, identifier)
);

-- 8. annotation
CREATE TABLE IF NOT EXISTS annotation (
    id INTEGER PRIMARY KEY,
    fragment_id INTEGER NOT NULL REFERENCES fragment(id),
    code_id INTEGER NOT NULL REFERENCES code(id),
    source TEXT NOT NULL DEFAULT 'llm' CHECK (source IN ('llm', 'human')),
    run_id INTEGER REFERENCES run(id),
    confidence REAL,
    rationale TEXT,
    quote TEXT NOT NULL,
    quote_start INTEGER,
    quote_end INTEGER,
    status TEXT NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed', 'accepted', 'edited', 'rejected'))
);

-- 9. review
CREATE TABLE IF NOT EXISTS review (
    id INTEGER PRIMARY KEY,
    annotation_id INTEGER NOT NULL REFERENCES annotation(id),
    reviewer TEXT,
    decision TEXT NOT NULL CHECK (decision IN ('approved', 'rejected', 'needs_revision')),
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 10. run
CREATE TABLE IF NOT EXISTS run (
    id INTEGER PRIMARY KEY,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model TEXT,
    prompt_hash TEXT,
    codebook_version_id INTEGER REFERENCES codebook_version(id),
    code_version TEXT,
    params_json TEXT
);

-- 11. theme
CREATE TABLE IF NOT EXISTS theme (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id INTEGER REFERENCES theme(id),
    level INTEGER DEFAULT 0
);

-- 12. theme_code
CREATE TABLE IF NOT EXISTS theme_code (
    id INTEGER PRIMARY KEY,
    theme_id INTEGER NOT NULL REFERENCES theme(id),
    code_id INTEGER NOT NULL REFERENCES code(id),
    UNIQUE(theme_id, code_id)
);

-- 13. embedding
CREATE TABLE IF NOT EXISTS embedding (
    id INTEGER PRIMARY KEY,
    fragment_id INTEGER NOT NULL REFERENCES fragment(id) UNIQUE,
    vector BLOB NOT NULL,
    model TEXT
);

-- 14. code_proposal
CREATE TABLE IF NOT EXISTS code_proposal (
    id INTEGER PRIMARY KEY,
    code_name TEXT NOT NULL,
    definition TEXT,
    reason TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    run_id INTEGER REFERENCES run(id)
);

-- 15. theme_proposal
CREATE TABLE IF NOT EXISTS theme_proposal (
    id INTEGER PRIMARY KEY,
    theme_name TEXT NOT NULL,
    reason TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    run_id INTEGER REFERENCES run(id)
);
