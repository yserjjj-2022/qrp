"""Рендер HTML-отчёта (M5).

Один файл: инлайн-стили и SVG, без JavaScript и внешних ресурсов. Рендер не
считает чисел: всё приходит в модели `InterviewReport` (SPEC-M6, «M5. Отчёт»).
Порядок блоков обязателен: происхождение → числа → дерево с кривыми → цитаты
→ ограничения.
"""

from __future__ import annotations

from html import escape

from qa.pipeline.aggregate import STATUS_LABELS, CoverageReport, ThemeCoverage
from qa.report.model import InterviewReport

_CSS = """
:root{--bg:#f7f8fa;--card:#fff;--ink:#1d2433;--muted:#5b6472;--line:#dde1e8;
--accent:#2f5fd0}
*{box-sizing:border-box}
body{margin:0;padding:36px 24px 72px;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:27px;line-height:1.25;margin:0 0 6px}
h2{font-size:18px;margin:0 0 14px}
h3{font-size:15px;margin:18px 0 8px}
.sub{color:var(--muted);margin:0 0 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:20px 22px 18px;margin:0 0 22px}
.banner{border:1px dashed #b9a14a;background:#fffbea;color:#6b5510;
border-radius:8px;padding:10px 14px;margin:0 0 22px;font-size:14px}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line);
vertical-align:top;font-size:14px}
th{font-size:12px;letter-spacing:.03em;text-transform:uppercase;color:var(--muted);
font-weight:600;border-bottom:1px solid #cfd5de}
tbody tr.theme td:first-child{padding-left:24px}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.list{margin:0;padding:0;list-style:none}
.list li{display:flex;justify-content:space-between;gap:16px;padding:6px 0;
border-bottom:1px solid #eef1f5}
.list li:last-child{border-bottom:0}
.list .v{color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.meta{display:flex;flex-wrap:wrap;gap:8px 26px;color:var(--muted);font-size:13px}
.meta b{color:var(--ink);font-weight:600}
blockquote{margin:0 0 8px;padding:8px 12px;border-left:3px solid #cfd5de;
background:#fafbfc;color:#333c4c;font-size:13px}
blockquote .src{display:block;color:var(--muted);font-size:11px;margin-top:4px}
.lim{margin:0;padding-left:18px}
.lim li{margin:0 0 8px;font-size:14px}
.note{color:var(--muted);font-size:13px;margin:10px 0 0}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:12px;padding:2px 8px;border-radius:6px;background:#f2f4f8;color:#333c4c}
.family{color:var(--muted);font-size:12px;font-weight:600;margin:12px 0 4px}
svg.spark{display:block}
"""


def _pct(part: int, total: int) -> str:
    """Процент с одним знаком; ноль знаменателя даёт 0 %."""
    if total <= 0:
        return "0 %"
    return f"{part / total * 100:.1f} %".replace(".", ",")


def _spark(cumulative: tuple[int, ...], width: int = 96, height: int = 24) -> str:
    """Мини-SVG накопления различных кодов по порядку реплик."""
    if not cumulative:
        return (
            f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}"'
            ' role="img" aria-label="нет накопления">'
            f'<line x1="1" y1="{height - 2}" x2="{width - 1}" y2="{height - 2}"'
            ' stroke="#c9cfd9" stroke-width="1.6" stroke-dasharray="3 3"/></svg>'
        )
    peak = max(cumulative)
    if len(cumulative) == 1:
        y = height - 2 - (cumulative[0] / peak) * (height - 4) if peak else height - 2
        return (
            f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}"'
            ' role="img" aria-label="накопление кодов">'
            f'<circle cx="{width / 2:.0f}" cy="{y:.0f}" r="2.6" fill="#2f5fd0" opacity="0.7"/></svg>'
        )
    step = (width - 2) / (len(cumulative) - 1)
    points = " ".join(
        f"{1 + index * step:.0f},{height - 2 - (value / peak) * (height - 4):.0f}"
        for index, value in enumerate(cumulative)
    )
    return (
        f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}"'
        ' role="img" aria-label="накопление кодов">'
        f'<polyline points="{points}" fill="none" stroke="#2f5fd0" stroke-width="1.6" opacity="0.7"/></svg>'
    )


def _provenance_block(report: InterviewReport) -> str:
    """Блок 1: происхождение чисел и список прогонов."""
    provenance = report.provenance
    meta = [
        ("Интервью", provenance.interview_code),
        ("Группа", provenance.group or "—"),
        ("Тип расшифровки", provenance.transcript_type),
        ("L2", f"v{provenance.l2_version}" if provenance.l2_version is not None else "—"),
        (
            "Кодбук",
            f"v{provenance.codebook_version_id}" if provenance.codebook_version_id is not None else "—",
        ),
        ("Снимок", provenance.snapshot_date),
        (
            "Прогон снимка",
            f"run {provenance.snapshot_run_id}" if provenance.snapshot_run_id is not None else "—",
        ),
    ]
    meta_html = "".join(f"<span>{escape(name)}: <b>{escape(value)}</b></span>" for name, value in meta)
    if provenance.runs:
        rows = "".join(
            "<tr>"
            f"<td>{run.id}</td><td>{escape(run.stage)}</td>"
            f"<td>{escape(run.model or '—')}</td>"
            f"<td>{escape((run.prompt_hash or '—')[:12])}</td>"
            f"<td>{escape(run.code_version or '—')}</td>"
            f"<td>{escape(run.started_at or '—')}</td>"
            "</tr>"
            for run in provenance.runs
        )
        runs_html = (
            "<h3>Прогоны, затронувшие интервью</h3><table><thead><tr>"
            "<th>run</th><th>стадия</th><th>модель</th><th>хеш промпта</th>"
            "<th>версия кода</th><th>начат</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    else:
        runs_html = '<p class="note">Прогонов, затронувших интервью, в БД нет.</p>'
    return f'<div class="card"><h2>1. Происхождение</h2><div class="meta">{meta_html}</div>{runs_html}</div>'


def _numbers_block(report: InterviewReport) -> str:
    """Блок 2: числа покрытия и качества данных."""
    coverage = report.coverage
    quality = coverage.quality
    status = quality.by_status
    left: list[tuple[str, str]] = [
        ("Аннотации, кроме отклонённых", str(coverage.annotations_total)),
        ("Из них проверено вручную", str(coverage.annotations_verified)),
        ("Статус accepted", str(status.get("accepted", 0))),
        ("Статус edited", str(status.get("edited", 0))),
        ("Статус proposed", str(status.get("proposed", 0))),
        ("Фрагментов всего", str(coverage.fragments_total)),
        ("Фрагментов с аннотацией", str(coverage.covered_fragments)),
        ("Фрагменты: coded", str(quality.coded)),
        ("Фрагменты: unmarked", str(quality.unmarked)),
        ("Фрагменты: skipped_garbled", str(quality.skipped_garbled)),
    ]
    right: list[tuple[str, str]] = [(f"Надёжность {level}", str(count)) for level, count in quality.reliability.items()]
    right.extend(
        [
            ("Аннотации с кодом OTH", str(quality.oth)),
            ("interpretation_only", str(quality.interpretation_only)),
            ("Фрагменты doubtful", str(quality.doubtful)),
            ("Аннотации на doubtful", f"{quality.accepted_on_doubtful} из {coverage.annotations_total}"),
            (
                "doubtful-фрагменты с аннотацией",
                f"{quality.fragments_with_accepted_doubtful} из {quality.doubtful}",
            ),
            ("Коды кодбука", f"{coverage.codebook.used} из {coverage.codebook.total}"),
            ("Вопросы без ответа", ", ".join(quality.questions_without_answer) or "нет"),
        ]
    )

    def items(rows: list[tuple[str, str]]) -> str:
        return "".join(
            f'<li><span>{escape(name)}</span><span class="v">{escape(value)}</span></li>' for name, value in rows
        )

    return (
        '<div class="card"><h2>2. Числа</h2>'
        f'<p class="note" style="margin:0 0 12px">Снимок на {escape(report.provenance.snapshot_date)}, '
        f"run {escape(str(report.provenance.snapshot_run_id))}. "
        f"Разметка модели; {coverage.annotations_verified} из {coverage.annotations_total} аннотаций "
        "проверены вручную, остальные не проверялись.</p>"
        "<table><tbody><tr>"
        f'<td style="width:50%;padding:0 18px 0 0;border:0"><ul class="list">{items(left)}</ul></td>'
        f'<td style="width:50%;padding:0;border:0"><ul class="list">{items(right)}</ul></td>'
        "</tr></tbody></table>"
        f"{_codebook_block(coverage)}"
        "</div>"
    )


def _codebook_block(coverage: CoverageReport) -> str:
    """Список неиспользованных кодов одним перечнем по семьям."""
    unused_total = sum(len(entry.names) for entry in coverage.codebook.unused)
    groups = "".join(
        f'<p class="family">{escape(entry.family)}</p>'
        '<div class="chips">' + "".join(f'<span class="chip">{escape(name)}</span>' for name in entry.names) + "</div>"
        for entry in coverage.codebook.unused
    )
    if not groups:
        groups = '<p class="note" style="margin:0">Все коды кодбука использованы.</p>'
    return (
        "<h3>Покрытие кодбука</h3>"
        f'<p class="note" style="margin:0 0 8px">Использовано {coverage.codebook.used} из '
        f"{coverage.codebook.total} кодов, не использовано {unused_total}. "
        "Список один, без деления по группам интервью.</p>"
        f"{groups}"
    )


def _tree_block(report: InterviewReport) -> str:
    """Блок 3: дерево тем с покрытием и кривыми накопления."""
    coverage = report.coverage
    rows: list[str] = []
    for node in coverage.nodes:
        rows.append(f'<tr><td colspan="5" style="background:#f2f4f8;font-weight:600">{escape(node.node)}</td></tr>')
        if not node.themes:
            rows.append('<tr><td colspan="5" class="note">Узел без тем.</td></tr>')
        for theme in node.themes:
            rows.append(_theme_row(theme, coverage.fragments_total))
    if coverage.annotations_total == 0:
        return (
            '<div class="card"><h2>3. Дерево тем</h2>'
            '<p class="note" style="margin:0">Аннотаций нет: дерево пусто.</p></div>'
        )
    return (
        '<div class="card"><h2>3. Дерево тем</h2><table><thead><tr>'
        '<th style="width:34%">Узел / тема</th><th class="num" style="width:12%">Аннотаций</th>'
        '<th class="num" style="width:13%">из них проверено</th>'
        '<th style="width:22%">Покрытие фрагментов</th><th style="width:19%">Кривая</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        '<p class="note">Статусы и кривые — эвристика на одном интервью; кривая показывает '
        "накопление различных кодов по порядку реплик.</p></div>"
    )


def _theme_row(theme: ThemeCoverage, fragments_total: int) -> str:
    """Строка темы: число аннотаций, проверенных, покрытие и кривая."""
    return (
        '<tr class="theme">'
        f"<td>{escape(theme.theme)} "
        f'<span class="note" style="display:inline">{escape(STATUS_LABELS.get(theme.status, theme.status))}</span></td>'
        f'<td class="num">{theme.annotations}</td>'
        f'<td class="num">{theme.verified}</td>'
        f"<td>{theme.covered_fragments} ({_pct(theme.covered_fragments, fragments_total)})</td>"
        f"<td>{_spark(theme.cumulative)}</td>"
        "</tr>"
    )


def _quotes_block(report: InterviewReport) -> str:
    """Блок 4: цитаты по темам, только из аннотаций интервью."""
    parts: list[str] = []
    for node in report.coverage.nodes:
        for theme in node.themes:
            if not theme.quotes:
                continue
            quotes = "".join(
                f"<blockquote>«{escape(quote.text)}»"
                f'<span class="src">код {escape(quote.code_name)}'
                + (f" · вопрос {escape(quote.question_identifier)}" if quote.question_identifier else "")
                + "</span></blockquote>"
                for quote in theme.quotes
            )
            parts.append(f"<h3>{escape(node.node)} → {escape(theme.theme)}</h3>{quotes}")
    if not parts:
        return (
            '<div class="card"><h2>4. Цитаты</h2>'
            '<p class="note" style="margin:0">Цитат нет: аннотаций по темам нет.</p></div>'
        )
    return (
        '<div class="card"><h2>4. Цитаты</h2>'
        + "".join(parts)
        + '<p class="note">Цитаты — срезы текста L2; вырезка отмечена многоточием.</p></div>'
    )


def _limitations_block(report: InterviewReport) -> str:
    """Блок 5: ограничения отчёта."""
    items = "".join(f"<li>{escape(item.text)}</li>" for item in report.limitations)
    return f'<div class="card"><h2>5. Ограничения</h2><ul class="lim">{items}</ul></div>'


def render_report(report: InterviewReport) -> str:
    """Собрать HTML-отчёт.

    Args:
        report: модель отчёта (происхождение, покрытие, ограничения).

    Returns:
        HTML-документ одной строкой.
    """
    return (
        '<!DOCTYPE html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
        f"<title>Отчёт по интервью — {escape(report.provenance.interview_code)}</title>\n"
        f'<style>{_CSS}</style>\n</head>\n<body>\n<div class="wrap">\n'
        "<h1>Отчёт по интервью</h1>\n"
        '<p class="sub">Числа и цитаты из базы; текст отчёта — детерминированный шаблон.</p>\n'
        f'<div class="banner">Разметка модели; {report.annotations_verified} из '
        f"{report.annotations_total} аннотаций проверены вручную, остальные не проверялись.</div>\n"
        f"{_provenance_block(report)}\n"
        f"{_numbers_block(report)}\n"
        f"{_tree_block(report)}\n"
        f"{_quotes_block(report)}\n"
        f"{_limitations_block(report)}\n"
        '<p class="note">Статический HTML без JavaScript и внешних ресурсов. '
        "Отчёт описывает одно интервью и не делает выводов о группе.</p>\n"
        "</div>\n</body>\n</html>\n"
    )
