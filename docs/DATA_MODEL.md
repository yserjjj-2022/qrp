# DATA_MODEL: схема базы данных

Все поля и связи фиксируются миграциями в `migrations/`:
`0001_create_tables.sql` (каркас), `0002_transcript_turns.sql` (записи и реплики),
`0003_workbook.sql` (рабочая книга, кодбук, лексикон, золотой набор, факты).

## Таблицы

### interview
Хранит информацию об интервью.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| id_external | TEXT NOT NULL UNIQUE | Внешний идентификатор |
| name | TEXT | Заголовок транскрипта |
| respondent_code | TEXT | Код респондента (R01...) — миграция 0002 |
| group | TEXT | Группа (Все / ТОРГ / Не торгующие) — миграция 0002 |
| transcript_source | TEXT | Источник/тип расшифровки — миграция 0002 |
| interview_date | TEXT | Дата интервью из заголовка или листа «Интервью» — миграция 0003 |
| created_at | TIMESTAMP | Дата создания |

### turn
Запись транскрипта (строка формата `ЧЧ:ММ:СС`, схема после миграции 0002).

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| interview_id | INTEGER NOT NULL | Ссылка на interview |
| line_no | INTEGER NOT NULL | Номер строки в исходном файле (L0) |
| start_sec | INTEGER NOT NULL | Метка начала реплики в секундах |
| speaker_role | TEXT NOT NULL | INTERVIEWER / RESPONDENT / OBSERVER |
| speaker_label | TEXT | Метка говорящего из транскрипта (миграция 0004) |
| text | TEXT NOT NULL | Текст L1 (псевдонимизированный) |
| pii_flag | INTEGER NOT NULL DEFAULT 0 | Совпадение с шаблонами PII (на ручную проверку) |

### utterance
Сборная реплика: объединение подряд идущих записей одного говорящего (миграция 0002).

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| interview_id | INTEGER NOT NULL | Ссылка на interview |
| speaker_role | TEXT NOT NULL | INTERVIEWER / RESPONDENT / OBSERVER |
| text | TEXT NOT NULL | Текст L1 (объединение записей без вставок) |
| turn_ids | TEXT NOT NULL | JSON-массив id записей turn |
| is_backchannel | INTEGER NOT NULL DEFAULT 0 | Короткая вставка-подтверждение («угу» и т. п.) |

### fragment
Фрагмент текста (сегмент для разметки).

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| turn_id | INTEGER NOT NULL | Ссылка на turn |
| text | TEXT NOT NULL | Текст фрагмента |
| start_pos | INTEGER | Позиция начала |
| end_pos | INTEGER | Позиция конца |
| created_at | TIMESTAMP | Дата создания |

### codebook_version
Версия кодбука.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| file_hash | TEXT NOT NULL | SHA-256 Excel-файла |
| extensions_hash | TEXT | Хеш расширений кодбука |
| workbook_version_id | INTEGER | Ссылка на версию рабочей книги — миграция 0003 |
| codebook_hash | TEXT | Хеш только кодбуковых листов, уникален — миграция 0003 |
| created_at | TIMESTAMP | Дата создания |

### code
Код из кодбука.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| codebook_version_id | INTEGER NOT NULL | Ссылка на версию |
| family | TEXT | Семейство кодов |
| name | TEXT NOT NULL | Идентификатор кода |
| kind | TEXT | Тип кода из листа «Коды» — миграция 0003 |
| title | TEXT | Название кода — миграция 0003 |
| definition | TEXT | Определение |
| include | TEXT | Что включаем |
| exclude | TEXT | Что исключаем |
| tiebreak_rule | TEXT | Правило при сомнении — миграция 0003 |
| tiebreak | INTEGER | Приоритет при конфликте |
| applies_to | TEXT | К чему применяется |
| origin | TEXT NOT NULL | base или inductive |
| status | TEXT NOT NULL | active, merged, retired |
| supersedes_id | INTEGER | Ссылка на заменяемый код |

### attribute
Атрибут кода.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| code_id | INTEGER NOT NULL | Ссылка на код |
| name | TEXT NOT NULL | Имя атрибута |
| value | TEXT | Значение |

### question
Вопрос из гайда интервью.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| codebook_version_id | INTEGER NOT NULL | Ссылка на версию |
| identifier | TEXT NOT NULL | Идентификатор (1, 7Т, 14Т-б) |
| block | TEXT | Блок гайда — миграция 0003 |
| group | TEXT | Группа (Все / ТОРГ / Не торгующие) — миграция 0003 |
| wording | TEXT | Формулировка вопроса — миграция 0003 |
| goal | TEXT | Что хотим понять — миграция 0003 |
| notes | TEXT | Заметки — миграция 0003 |

### annotation
Разметка фрагмента.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| fragment_id | INTEGER NOT NULL | Ссылка на фрагмент |
| code_id | INTEGER NOT NULL | Ссылка на код |
| source | TEXT NOT NULL | llm или human |
| run_id | INTEGER | Ссылка на прогон |
| confidence | REAL | Уверенность |
| rationale | TEXT | Обоснование |
| quote | TEXT NOT NULL | Дословная цитата |
| quote_start | INTEGER | Позиция начала цитаты |
| quote_end | INTEGER | Позиция конца цитаты |
| status | TEXT NOT NULL | proposed, accepted, edited, rejected |

### review
Результат ревью аннотации.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| annotation_id | INTEGER NOT NULL | Ссылка на аннотацию |
| reviewer | TEXT | Идентификатор ревьюера |
| decision | TEXT NOT NULL | approved, rejected, needs_revision |
| comment | TEXT | Комментарий |
| created_at | TIMESTAMP | Дата создания |

### run
Прогон пайплайна.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| started_at | TIMESTAMP | Дата начала |
| model | TEXT | Идентификатор модели |
| prompt_hash | TEXT | Хеш промпта |
| codebook_version_id | INTEGER | Ссылка на версию кодбука |
| code_version | TEXT | Версия кода |
| params_json | TEXT | Параметры в JSON |

### theme
Тема из дерева тем.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| name | TEXT NOT NULL | Название темы |
| parent_id | INTEGER | Ссылка на родителя (self-ref) |
| level | INTEGER | Уровень дерева |

### theme_code
Связь темы с кодом.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| theme_id | INTEGER NOT NULL | Ссылка на тему |
| code_id | INTEGER NOT NULL | Ссылка на код |

### embedding
Векторная репрезентация фрагмента.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| fragment_id | INTEGER NOT NULL UNIQUE | Ссылка на фрагмент |
| vector | BLOB NOT NULL | Вектор (байты) |
| model | TEXT | Модель эмбеддингов |

### code_proposal
Предложение нового кода.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| code_name | TEXT NOT NULL | Имя предлагаемого кода |
| definition | TEXT | Определение |
| reason | TEXT | Обоснование |
| status | TEXT NOT NULL | pending, approved, rejected |
| run_id | INTEGER | Ссылка на прогон |

### theme_proposal
Предложение новой темы.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| theme_name | TEXT NOT NULL | Название предлагаемой темы |
| reason | TEXT | Обоснование |
| status | TEXT NOT NULL | pending, approved, rejected |
| run_id | INTEGER | Ссылка на прогон |

## Рабочая книга (миграция 0003)

### workbook_version
Прочитанная версия рабочей книги проекта.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| workbook_path | TEXT NOT NULL | Путь к файлу книги |
| file_hash | TEXT NOT NULL UNIQUE | SHA-256 файла |
| codebook_hash | TEXT NOT NULL | Хеш кодбуковых листов |
| loaded_at | TIMESTAMP | Когда загружена |

### workbook_sheet
Хеши листов в составе версии книги.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| workbook_version_id | INTEGER NOT NULL | Ссылка на версию книги |
| sheet_name | TEXT NOT NULL | Имя листа |
| header_hash | TEXT NOT NULL | Хеш заголовка |
| content_hash | TEXT NOT NULL | Хеш содержимого |
| n_rows | INTEGER NOT NULL DEFAULT 0 | Число строк |
| UNIQUE(workbook_version_id, sheet_name) | | |

### question_code
Связь «вопрос -> код» из листов «Вопросы_и_коды» и «Коды».

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| question_id | INTEGER NOT NULL | Ссылка на question |
| code_id | INTEGER NOT NULL | Ссылка на code |
| UNIQUE(question_id, code_id) | | |

### attribute_definition
Описание атрибута разметки из листа «Атрибуты».

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| codebook_version_id | INTEGER NOT NULL | Ссылка на версию |
| name | TEXT NOT NULL | Имя поля |
| kind | TEXT | Тип (атрибут / оценка) |
| description | TEXT | Описание значений |
| UNIQUE(codebook_version_id, name) | | |

### glossary_term
Словарь замен для интервьюера (лист «Словарь_замен»).

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| codebook_version_id | INTEGER NOT NULL | Ссылка на версию |
| term | TEXT NOT NULL | Не говорить |
| replacement | TEXT | Говорить |
| UNIQUE(codebook_version_id, term) | | |

### lexicon
Лексикон терминов: лист «Лексикон» плюс решения review-terms.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| canonical | TEXT NOT NULL UNIQUE | Каноническая форма |
| variants_json | TEXT NOT NULL DEFAULT '[]' | Варианты ASR (JSON) |
| definition | TEXT | Определение |
| status | TEXT NOT NULL DEFAULT 'proposed' | proposed / approved / rejected |
| source | TEXT NOT NULL DEFAULT 'workbook' | workbook или review (review приоритетнее) |
| updated_at | TIMESTAMP | Обновлено |

### uncertainty_marker
Маркеры неуверенности (лист «Маркеры_неуверенности» или умолчания config.yaml).

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| expression | TEXT NOT NULL UNIQUE | Выражение |
| note | TEXT | Примечание |
| source | TEXT NOT NULL DEFAULT 'workbook' | workbook или default |

### golden_sample
Золотой набор: реплики L1 и ручной L2 (лист «Золотой_набор» или xlsx экспорта).

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| interview_code | TEXT | Код интервью |
| utterance_ref | TEXT | Ссылка вида utterance:<id> |
| text_l1 | TEXT NOT NULL | Текст L1 |
| text_l2_manual | TEXT | Ручной L2 |
| source | TEXT NOT NULL DEFAULT 'workbook' | workbook / import |
| created_at | TIMESTAMP | |
| UNIQUE(interview_code, utterance_ref, text_l1) | | |

### reference_fact
Эталонные факты из листов с префиксом «Факты_».

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| workbook_version_id | INTEGER NOT NULL | Ссылка на версию книги |
| sheet_name | TEXT NOT NULL | Имя листа |
| external_id | TEXT NOT NULL | Идентификатор факта |
| question | TEXT | Вопрос сверки |
| answer | TEXT | Ответ |
| comment | TEXT | Комментарий |
| status | TEXT | Статус |
| UNIQUE(workbook_version_id, sheet_name, external_id) | | |

### interview_speaker
Говорящие интервью: метки, роли и число слов (init-interview).

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| interview_id | INTEGER NOT NULL | Ссылка на interview |
| speaker_label | TEXT NOT NULL | Метка из транскрипта |
| role | TEXT NOT NULL | INTERVIEWER / RESPONDENT / OBSERVER |
| words | INTEGER NOT NULL DEFAULT 0 | Число слов |
| UNIQUE(interview_id, speaker_label) | | |

Роли из листа «Роли_говорящих» применяются к `turn.speaker_role` и пересобирают `utterance`
(`load-project`). Пересборка запрещена, если на реплики ссылаются кандидаты `new_term`:
их ссылки потерялись бы, поэтому сначала закрывается `review-terms`.

### new_term
Кандидаты новых терминов для review-terms (используется в M1b).

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| interview_id | INTEGER | Ссылка на interview |
| surface_form | TEXT NOT NULL | Как встретилось в тексте |
| context_utterance_id | INTEGER | Ссылка на utterance |
| status | TEXT NOT NULL DEFAULT 'proposed' | proposed / approved / rejected |
| created_at | TIMESTAMP | |
| UNIQUE(surface_form, context_utterance_id) | | |

## Допустимые значения справочников (config.yaml)

### Группы (`interview.group`, `question.group`)
Канонические значения: `Все`, `ТОРГ`, `Не торгующие`.
Источник — секция `groups` в `config.yaml`: `canonical` плюс `aliases` для каждого значения.
Сравнение идёт по casefold со схлопыванием пробелов и подчёркиваний, поэтому
`не торгующие`, `не_торгующие`, `Не торгующие` дают одно значение.
Неизвестное значение не является ошибкой: оно сохраняется как есть (после `strip`),
а в лог и в отчёт выводится предупреждение. Это позволяет загрузить книгу
с новой группой до того, как оператор дополнит `config.yaml`.

### Статусы лексикона (`lexicon.status`)
Внутренние значения: `proposed`, `approved`, `rejected`.
Приведение задаётся секцией `lexicon_status_map` в `config.yaml`
(`предложен` -> `proposed`, `утверждён` -> `approved`, `отклонён` -> `rejected`);
английские значения принимаются как есть, регистр не учитывается.
Пустой статус считается `proposed`. Неизвестное значение не ошибка: сохраняется
`proposed`, а в отчёт `load-project` попадает предупреждение с номером строки листа.
Решения `review-terms` (`lexicon.source = 'review'`) приоритетнее листа «Лексикон»
и не перезаписываются.
