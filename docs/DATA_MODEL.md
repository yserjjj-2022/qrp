# DATA_MODEL: схема базы данных

Все поля и связи фиксируются в миграции `migrations/0001_create_tables.sql`.

## Таблицы

### interview
Хранит информацию об интервью.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| id_external | TEXT NOT NULL UNIQUE | Внешний идентификатор |
| name | TEXT | Имя/описание |
| created_at | TIMESTAMP | Дата создания |

### turn
Реплика (поворот) в разговоре.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| interview_id | INTEGER NOT NULL | Ссылка на interview |
| speaker | TEXT | Идентификатор спикера (код) |
| text | TEXT NOT NULL | Текст реплики |
| question_id | INTEGER | Ссылка на вопрос |
| created_at | TIMESTAMP | Дата создания |

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
| created_at | TIMESTAMP | Дата создания |

### code
Код из кодбука.

| Поле | Тип | Описание |
|------|-----|----------|
| id | INTEGER PRIMARY KEY | Автоинкремент |
| codebook_version_id | INTEGER NOT NULL | Ссылка на версию |
| family | TEXT | Семейство кодов |
| name | TEXT NOT NULL | Идентификатор кода |
| definition | TEXT | Определение |
| include | TEXT | Что включаем |
| exclude | TEXT | Что исключаем |
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
