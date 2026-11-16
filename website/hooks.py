"""MkDocs hook: generate the week and guide pages from the lab repository.

The lab READMEs stay the single source of truth. Before every build this hook
copies them into ``docs/weeks`` and ``docs/guides`` and rewrites their links:

* links to other lab READMEs or course docs point to the matching site page;
* links to notebooks, scripts, fixtures and configs point to GitHub at
  ``extra.source_ref`` (set ``DOCS_SOURCE_REF`` to change it);
* the Google Sheets registration link is replaced with plain text, because the
  site is public.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = Path(__file__).resolve().parent / "docs"

RELEASED_WEEKS = range(1, 9)  # Weeks 1-8 have lab content; 9-15 are previews.

GUIDES = {
    "SLURM-QUICK-GUIDE.md": "guides/slurm.md",
    "AGENTS.md": "guides/workspace-rules.md",
    "docs/DATA-CONTRACT.md": "guides/data-contract.md",
    "docs/PROJECT-CONTRACT.md": "guides/project-contract.md",
    "docs/MIDTERM-PRESENTATION.md": "guides/midterm.md",
}

LINK = re.compile(r"(!?\[[^\]]*\])\(([^)\s]+)(\s+\"[^\"]*\")?\)")
SHEETS = re.compile(r"\[([^\]]+)\]\(https://docs\.google\.com/spreadsheets/[^)]+\)")
NAV_LINE = re.compile(r"^\[← Back to course README\].*$", re.MULTILINE)
GITHUB_SLURM = "https://github.com/ntcuong2103/project-ws2026/blob/main/SLURM-QUICK-GUIDE.md"


def _week_dirs() -> dict[int, str]:
    weeks = {}
    for path in sorted((REPO_ROOT / "labs").glob("week-*")):
        number = int(path.name.split("-")[1])
        weeks[number] = path.name
    return weeks


def _page_map() -> dict[str, str]:
    """Repo-relative source path -> site page path (relative to docs/)."""
    pages = dict(GUIDES)
    for number, folder in _week_dirs().items():
        page = f"weeks/week-{number:02d}.md"
        pages[f"labs/{folder}"] = page
        pages[f"labs/{folder}/README.md"] = page
    pages["README.md"] = "syllabus.md"
    pages["labs"] = "weeks/index.md"
    return pages


def _rewrite(text: str, source: str, page: str, repo: str, ref: str, pages: dict[str, str]) -> str:
    source_dir = PurePosixPath(source).parent
    page_dir = PurePosixPath(page).parent

    def site_link(target_page: str, anchor: str) -> str:
        rel = os.path.relpath(target_page, page_dir.as_posix() or ".")
        return rel + anchor

    def replace(match: re.Match[str]) -> str:
        label, target, title = match.group(1), match.group(2), match.group(3) or ""
        if target == GITHUB_SLURM:
            return f"{label}({site_link(pages['SLURM-QUICK-GUIDE.md'], '')}{title})"
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            return match.group(0)
        path, _, anchor = target.partition("#")
        anchor = f"#{anchor}" if anchor else ""
        resolved = os.path.normpath((source_dir / path).as_posix()).replace(os.sep, "/").rstrip("/")
        if resolved in pages:
            # Anchors into the long course README do not survive the move.
            return f"{label}({site_link(pages[resolved], '' if resolved == 'README.md' else anchor)}{title})"
        kind = "tree" if (REPO_ROOT / resolved).is_dir() or path.endswith("/") else "blob"
        return f"{label}({repo}/{kind}/{ref}/{resolved}{anchor}{title})"

    out, in_fence = [], False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        out.append(line if in_fence else LINK.sub(replace, SHEETS.sub(r"\1 (link shared in class)", line)))
    return "".join(out)


def _write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def _source_note(source: str, repo: str, ref: str) -> str:
    return (
        f"\n\n---\n\n<small>Generated from [`{source}`]({repo}/blob/{ref}/{source}). "
        "Edit the lab repository, not this page.</small>\n"
    )


def _schedule_rows(readme: str) -> list[list[str]]:
    match = re.search(r"## Tentative 15-Week Schedule\n(.*?)\n## ", readme, re.DOTALL)
    if not match:
        raise RuntimeError("Could not find the schedule table in README.md")
    rows = []
    for line in match.group(1).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if line.startswith("|") and cells[0].isdigit():
            rows.append(cells)
    if len(rows) != 15:
        raise RuntimeError(f"Expected 15 schedule rows in README.md, found {len(rows)}")
    return rows


def _schedule_plan(rows: list[list[str]], extra: dict) -> str:
    """Render the README schedule as stacked blocks; a 6-column table needs side-scrolling."""
    blocks = []
    for number, phase, topic, work, deliverable, *_ in rows:
        n = int(number)
        title, _blurb = WEEKS[n]
        status, label = _week_status(n, extra)
        blocks.append(
            f'<div class="wk-plan wk-plan--{_phase(title)}" markdown>\n'
            f'<p class="wk-plan__head"><a href="week-{n:02d}/">'
            f'<span class="wk-plan__num">Week {n:02d}</span> {phase}</a> '
            f'<span class="wk-status wk-status--{status}">{label}</span></p>\n\n'
            f"{topic}\n\n"
            f"**In the lab:** {work}\n\n"
            f"**Deliverable:** {deliverable}\n\n"
            "</div>\n"
        )
    return "\n".join(blocks)


WEEKS = {
    1: ("Orientation", "Kickoff, the source paper, server access, and team formation"),
    2: ("Foundations", "Problem framing, NomNaOCR audit, and edit distance"),
    3: ("Detection I", "Single-class YOLO11 character detection on MTH pages"),
    4: ("Detection II", "Fine-tune on Nakagawa, evaluate on NomNaOCR"),
    5: ("Recognition I", "IDS decomposition, vocabulary coverage, and crops"),
    6: ("Recognition II", "Inside the recognizer: logits and forward passes"),
    7: ("Recognition III", "Bidirectional decoding and the full pipeline demo"),
    8: ("Midterm", "15-minute talks on detection and recognition"),
    9: ("Alignment I", "Assign detected boxes to transcription lines"),
    10: ("Alignment II", "Two-level IDS dynamic-programming alignment"),
    11: ("Alignment III", "Confidence scores against the binary-flag baseline"),
    12: ("Pseudo-labeling I", "The confidence-weighted fine-tuning loop"),
    13: ("Pseudo-labeling II", "Modified CER and precision by confidence band"),
    14: ("Integration", "Robustness to detector noise and the final dry run"),
    15: ("Final", "Final presentations, written report, and code"),
}


def _phase(title: str) -> str:
    return title.split()[0].lower()


def _week_status(number: int, extra: dict) -> tuple[str, str]:
    """(css modifier, label) for a week given the course's current position."""
    current = int(extra.get("current_week", 1))
    if number == 8:
        return "midterm", extra.get("midterm_label", "Midterm")
    if number < current:
        return "done", "Done"
    if number == current:
        return "next", extra.get("current_label", "Up next")
    return "planned", "Planned" if number not in RELEASED_WEEKS else "Released"


def on_config(config):  # noqa: ANN001, ANN201 - MkDocs hook signature
    extra = config["extra"]
    weeks = []
    for number, (title, blurb) in WEEKS.items():
        status, label = _week_status(number, extra)
        weeks.append({
            "number": number, "title": title, "blurb": blurb, "phase": _phase(title),
            "status": status, "label": label, "url": f"weeks/week-{number:02d}/",
        })
    extra["weeks"] = weeks
    return config


def _week_banner(number: int, extra: dict) -> str:
    title, blurb = WEEKS[number]
    status, label = _week_status(number, extra)
    return (
        f'<div class="wk-banner wk-banner--{_phase(title)}">'
        f'<span class="wk-banner__num">Week {number:02d}</span>'
        f'<span class="wk-banner__phase">{title}</span>'
        f'<span class="wk-status wk-status--{status}">{label}</span>'
        f'<span class="wk-banner__blurb">{blurb}</span>'
        "</div>\n\n"
    )


def on_pre_build(config) -> None:  # noqa: ANN001 - MkDocs hook signature
    repo = config["extra"]["source_repo"]
    ref = config["extra"]["source_ref"] or "main"
    pages = _page_map()

    for number, folder in _week_dirs().items():
        source = f"labs/{folder}/README.md"
        page = f"weeks/week-{number:02d}.md"
        text = NAV_LINE.sub("", (REPO_ROOT / source).read_text(encoding="utf-8"))
        text = _rewrite(text, source, page, repo, ref, pages)
        if number not in RELEASED_WEEKS:
            heading_end = text.find("\n") + 1
            text = (
                text[:heading_end]
                + '\n!!! info "Coming soon"\n    This is the planned scope from the course plan. '
                "The full lab handout will be published before the session.\n"
                + text[heading_end:]
            )
        text = _week_banner(number, config["extra"]) + text
        _write_if_changed(DOCS / page, text + _source_note(source, repo, ref))

    for source, page in GUIDES.items():
        text = (REPO_ROOT / source).read_text(encoding="utf-8")
        text = _rewrite(text, source, page, repo, ref, pages)
        _write_if_changed(DOCS / page, text + _source_note(source, repo, ref))

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    plan = _rewrite(
        _schedule_plan(_schedule_rows(readme), config["extra"]), "README.md", "weeks/index.md", repo, ref, pages
    )
    cards = "".join(
        f'<a class="wk-card wk-card--{w["phase"]} wk-card--{w["status"]}" href="week-{w["number"]:02d}/">'
        f'<span class="wk-card__top"><span class="wk-card__num">W{w["number"]:02d}</span>'
        f'<span class="wk-status wk-status--{w["status"]}">{w["label"]}</span></span>'
        f'<span class="wk-card__title">{w["title"]}</span>'
        f'<span class="wk-card__blurb">{w["blurb"]}</span></a>'
        for w in config["extra"]["weeks"]
    )
    index = (
        "# Weekly schedule\n\n"
        "Each session is 4 academic hours: a 5-minute review per group of the previous week's "
        "deliverable, then the week's lab. Weeks 8 and 15 are presentations.\n\n"
        '!!! tip "Where we are"\n'
        "    Weeks 1–6 are done. Classes resume in December with Week 7, followed by the "
        "Week 8 midterm on **Tuesday 15 December 2026**. Weeks 1–7 labs are available as "
        "practice during the break.\n\n"
        f'<div class="wk-grid">{cards}</div>\n\n'
        "## Detailed plan\n\n"
        f"{plan}\n"
    )
    _write_if_changed(DOCS / "weeks/index.md", index + _source_note("README.md", repo, ref))
