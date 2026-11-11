"""Рендер HTML-дашборда покрытия дерева тем (M4).

Один файл: инлайн-стили и SVG, без JavaScript и внешних ресурсов
(SPEC-M6, «Макет дашборда»). Рендер не считает чисел — всё приходит в модели
`CoverageReport`, поэтому в модуле нет ни SQL, ни арифметики показателей.
"""

from __future__ import annotations

from html import escape

from qa.pipeline.aggregate import (
    HEURISTIC_CAPTION,
    STATUS_LABELS,
    CodebookCoverage,
    CoverageReport,
    DataQuality,
    ThemeCoverage,
)

STATUS_CSS = {
    "no_data": "none",
    "few": "few",
    "has_data": "has",
}

_CSS = """
:root{--bg:#f7f8fa;--card:#fff;--ink:#1d2433;--muted:#5b6472;--line:#dde1e8;
--none:#eceff3;--none-ink:#5b6472;--few:#fdf3d7;--few-ink:#7a5b00;
--has:#e6f4ea;--has-ink:#1e6b36;--accent:#2f5fd0}
*{box-sizing:border-box}
body{margin:0;padding:32px 24px 64px;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1120px;margin:0 auto}
h1{font-size:26px;line-height:1.25;margin:0 0 6px}
h2{font-size:18px;margin:0 0 14px}
.sub{color:var(--muted);margin:0 0 22px}
.banner{border:1px dashed #b9a14a;background:#fffbea;color:#6b5510;
border-radius:8px;padding:10px 14px;margin:0 0 26px;font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:20px 20px 18px;margin:0 0 22px}
.meta{display:flex;flex-wrap:wrap;gap:8px 28px;color:var(--muted);font-size:13px;margin:0 0 18px}
.meta b{color:var(--ink);font-weight:600}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);
vertical-align:middle;font-size:14px}
th{font-size:12px;letter-spacing:.03em;text-transform:uppercase;color:var(--muted);
font-weight:600;border-bottom:1px solid #cfd5de}
tbody tr.node td{background:#f2f4f8;font-weight:600}
tbody tr.node td:first-child{padding-left:14px}
tbody tr.theme td:first-child{padding-left:30px}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;
font-weight:600;white-space:nowrap}
.pill.none{background:var(--none);color:var(--none-ink)}
.pill.few{background:var(--few);color:var(--few-ink)}
.pill.has{background:var(--has);color:var(--has-ink)}
.heur{color:var(--muted);font-size:11px;display:block;margin-top:2px}
.heur-inline{color:var(--muted);font-size:11px}
.codes{font-variant-numeric:tabular-nums}
.codes .of{color:var(--muted)}
.bar{display:inline-block;height:6px;border-radius:3px;background:var(--accent);
vertical-align:middle;margin-right:6px;opacity:.55}
details{margin:2px 0 10px 30px}
details summary{cursor:pointer;color:var(--accent);font-size:13px;margin:0 0 6px}
blockquote{margin:0 0 8px;padding:8px 12px;border-left:3px solid #cfd5de;
background:#fafbfc;color:#333c4c;font-size:13px}
blockquote .src{display:block;color:var(--muted);font-size:11px;margin-top:4px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:26px}
@media (max-width:860px){.cols{grid-template-columns:1fr}}
.list{margin:0;padding:0;list-style:none}
.list li{display:flex;justify-content:space-between;gap:12px;padding:6px 0;
border-bottom:1px solid #eef1f5;font-size:14px}
.list li:last-child{border-bottom:0}
.list .v{color:var(--muted);font-variant-numeric:tabular-nums}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:12px;padding:2px 8px;border-radius:6px;background:#f2f4f8;
color:#333c4c;font-variant-numeric:tabular-nums}
.chip.restricted{background:#f6f7f9;color:#8b94a3;border:1px dashed #c9cfd9}
.note{color:var(--muted);font-size:13px;margin:10px 0 0}
.kpi{display:flex;flex-wrap:wrap;gap:26px;margin:0 0 4px}
.kpi div{min-width:130px}
.kpi .k{font-size:24px;font-weight:650}
.kpi .l{font-size:12px;color:var(--muted)}
svg.spark{display:block}
"""


def _pct(part: int, total: int) -> str:
    """Процент с одним знаком; ноль знаменателя даёт 0 %."""
    if total <= 0:
        return "0 %"
    return f"{part / total * 100:.1f} %".replace(".", ",")


def _spark(cumulative: tuple[int, ...], width: int = 72, height: int = 22) -> str:
    """Мини-SVG накопления кодов по порядку реплик.

    Одно значение даёт точку на уровне своего числа: кривая из одного шага
    читалась бы как отсутствие данных, хотя код найден.
    """
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
            f'<circle cx="{width / 2:.0f}" cy="{y:.0f}" r="2.4" fill="#2f5fd0" opacity="0.7"/></svg>'
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


def _theme_rows(theme: ThemeCoverage, fragments_total: int) -> str:
    """Строка темы и её цитаты в раскрывающемся блоке."""
    pill = STATUS_CSS.get(theme.status, "none")
    label = STATUS_LABELS.get(theme.status, theme.status)
    covered = ""
    if theme.covered_fragments and fragments_total > 0:
        width = max(4, round(theme.covered_fragments / fragments_total * 60))
        covered = f'<span class="bar" style="width:{width}px"></span>'
    row = (
        '<tr class="theme">'
        f"<td>{escape(theme.theme)}</td>"
        f'<td><span class="pill {pill}">{escape(label)}</span>'
        f'<span class="heur">{escape(HEURISTIC_CAPTION)}</span></td>'
        f'<td class="num">{theme.annotations}</td>'
        f'<td class="codes num">{theme.distinct_codes} <span class="of">/ {theme.theme_codes}</span></td>'
        f"<td>{covered}{theme.covered_fragments} ({_pct(theme.covered_fragments, fragments_total)})</td>"
        f"<td>{_spark(theme.cumulative)}</td>"
        "</tr>"
    )
    if theme.quotes:
        quotes = "".join(
            f"<blockquote>«{escape(quote.text)}»"
            f'<span class="src">код {escape(quote.code_name)}'
            + (f" · вопрос {escape(quote.question_identifier)}" if quote.question_identifier else "")
            + "</span></blockquote>"
            for quote in theme.quotes
        )
        detail = f"<details><summary>цитаты ({len(theme.quotes)})</summary>{quotes}</details>"
    elif theme.theme_codes == 0:
        detail = '<p class="note" style="margin:2px 0 12px 30px">Тема без кодов: привязка к кодубуку пуста.</p>'
    else:
        detail = '<p class="note" style="margin:2px 0 12px 30px">Цитат нет: принятых аннотаций по теме нет.</p>'
    return row + f'<tr><td colspan="6" style="padding:0;border:0">{detail}</td></tr>'


def _codebook_block(codebook: CodebookCoverage) -> str:
    """Блок покрытия кодбука: один список с пометкой неприменимых кодов."""
    chips = "".join(
        f'<span class="chip{" restricted" if not entry.applicable else ""}">{escape(name)}</span>'
        for entry in codebook.unused
        for name in entry.names
    )
    unused_total = sum(len(entry.names) for entry in codebook.unused)
    if not chips:
        chips = '<p class="note" style="margin:0">Все коды кодбука использованы.</p>'
    return (
        '<div class="card"><h2>2. Покрытие кодбука</h2>'
        f'<p style="margin:0 0 4px"><b>Использовано {codebook.used} из {codebook.total} кодов.'
        f" Не использовано {unused_total}.</b> "
        f'<span class="heur-inline">{escape(HEURISTIC_CAPTION)}</span></p>'
        '<p class="note" style="margin-top:0">Один список без разделения по группам: пунктиром '
        "отмечены коды, чьи вопросы задаются только другой группе, — их отсутствие в этом "
        "интервью не дефект. Остальные могли встретиться, но принятых аннотаций нет.</p>"
        f'<div class="chips">{chips}</div>'
        f'<p class="note">Пунктиром: {codebook.unused_not_applicable} кодов не применимы к группе. '
        f"Сплошным: {codebook.unused_applicable} кодов применимы, но не найдены.</p>"
        "</div>"
    )


def _quality_block(quality: DataQuality) -> str:
    """Блок качества данных: статусы, надёжность, doubtful, вопросы без ответа."""
    status = quality.by_status
    left: list[tuple[str, str]] = [
        ("Аннотации: accepted", str(status.get("accepted", 0))),
        ("Аннотации: edited", str(status.get("edited", 0))),
        ("Аннотации: proposed (не учтены)", str(status.get("proposed", 0))),
        ("Фрагменты: coded", str(quality.coded)),
        ("Фрагменты: unmarked", str(quality.unmarked)),
        ("Фрагменты: skipped_garbled", str(quality.skipped_garbled)),
        ("Фрагменты: doubtful", str(quality.doubtful)),
        ("Аннотации с кодом OTH", str(quality.oth)),
    ]
    accepted = status.get("accepted", 0) + status.get("edited", 0)
    right: list[tuple[str, str]] = [(f"Надёжность {level}", str(count)) for level, count in quality.reliability.items()]
    right.append(("Принятые аннотации на doubtful", f"{quality.accepted_on_doubtful} из {accepted}"))
    right.append(
        ("doubtful-фрагменты с аннотацией", f"{quality.fragments_with_accepted_doubtful} из {quality.doubtful}")
    )
    right.append(
        (
            "Вопросы без ответа",
            ", ".join(quality.questions_without_answer) if quality.questions_without_answer else "нет",
        )
    )

    def items(rows: list[tuple[str, str]]) -> str:
        cells = (f'<li><span>{escape(name)}</span><span class="v">{escape(value)}</span></li>' for name, value in rows)
        return "".join(cells)

    return (
        '<div class="card"><h2>3. Качество данных</h2><div class="cols">'
        f'<div><ul class="list">{items(left)}</ul></div>'
        f'<div><ul class="list">{items(right)}</ul></div>'
        "</div></div>"
    )


def render_dashboard(report: CoverageReport, snapshot_date: str) -> str:
    """Собрать HTML дашборда покрытия.

    Args:
        report: модель покрытия интервью.
        snapshot_date: дата снимка чисел (ISO) — подписывает шапку.

    Returns:
        HTML-документ одной строкой.
    """
    meta = [
        ("Интервью", report.interview_code),
        ("Группа", report.group or "—"),
        ("Снимок", snapshot_date),
        ("Прогон", f"run {report.run_id}" if report.run_id is not None else "—"),
        ("L2", f"v{report.l2_version}" if report.l2_version is not None else "—"),
        ("Кодбук", f"v{report.codebook_version_id}" if report.codebook_version_id is not None else "—"),
    ]
    meta_html = "".join(f"<span>{escape(name)}: <b>{escape(value)}</b></span>" for name, value in meta)

    accepted = report.quality.by_status.get("accepted", 0) + report.quality.by_status.get("edited", 0)
    kpi = (
        f'<div><div class="k">{accepted}</div><div class="l">принятых аннотаций</div></div>'
        f'<div><div class="k">{report.covered_fragments} / {report.fragments_total}</div>'
        '<div class="l">фрагментов с аннотацией</div></div>'
        f'<div><div class="k">{report.codebook.used} / {report.codebook.total}</div>'
        '<div class="l">кодов кодбука</div></div>'
        f'<div><div class="k">{report.quality.by_status.get("proposed", 0)}</div>'
        '<div class="l">proposed, не учтены</div></div>'
    )

    rows: list[str] = []
    for node in report.nodes:
        rows.append(f'<tr class="node"><td>{escape(node.node)}</td><td colspan="5"></td></tr>')
        if not node.themes:
            rows.append(
                '<tr><td colspan="6" style="padding:0;border:0">'
                '<p class="note" style="margin:2px 0 12px 30px">Узел без тем.</p></td></tr>'
            )
        for theme in node.themes:
            rows.append(_theme_rows(theme, report.fragments_total))

    tree = (
        '<div class="card"><h2>1. Дерево тем</h2><table><thead><tr>'
        '<th style="width:34%">Узел / тема</th><th style="width:20%">Статус</th>'
        '<th class="num" style="width:9%">Аннотаций</th><th class="num" style="width:11%">Кодов</th>'
        '<th style="width:16%">Покрытие фрагментов</th><th style="width:10%">Кривая</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        '<p class="note">Статусы листьев — эвристика на одном интервью. Шкала 0 / 1–3 / 4 и больше '
        "отвечает только на вопрос «есть ли чем показать тему», а не на вопрос «раскрыта ли тема».</p>"
        "</div>"
    )

    return (
        '<!DOCTYPE html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
        f"<title>Дашборд покрытия тем — {escape(report.interview_code)}</title>\n"
        f'<style>{_CSS}</style>\n</head>\n<body>\n<div class="wrap">\n'
        "<h1>Дашборд покрытия дерева тем</h1>\n"
        '<p class="sub">Один вопрос: где нет данных по темам кодбука.</p>\n'
        f'<div class="banner">Снимок чисел на {escape(snapshot_date)}, '
        f"run {escape(str(report.run_id))}. В агрегаты входят только accepted и edited: "
        f"{report.quality.by_status.get('proposed', 0)} proposed не учтены.</div>\n"
        f'<div class="card"><div class="meta">{meta_html}</div><div class="kpi">{kpi}</div></div>\n'
        f"{tree}\n{_codebook_block(report.codebook)}\n"
        f"{_quality_block(report.quality)}\n"
        '<div class="card"><h2>4. Цитаты</h2><p class="note" style="margin:0">'
        "Цитаты раскрываются внутри блока 1, под строкой темы: так цитата остаётся рядом со "
        "счётчиками своей темы.</p></div>\n"
        '<p class="note">Статический HTML без JavaScript и внешних ресурсов.</p>\n'
        "</div>\n</body>\n</html>\n"
    )
