"""Generate the landing-page illustration from one real NomNaOCR page.

Source: nomnaocr/line_labels/DVSKTT-1 Quyen thu/DVSKTT_thu_III_2a.txt (the
preface of Đại Việt sử ký toàn thư). Each line is a column polygon plus its
transcription; columns run right to left. Characters are spaced evenly inside
each column and get an animated "detection" box. One character (記) is
highlighted and decomposed into its IDS, ⿰言己, as the recognizer sees it.

Run: uv run --project website python website/tools/make_manuscript_svg.py
"""

from pathlib import Path

LINES = [
    (257, 55, 279, 393, "火之後其書不傳盖成之至難意若有待"),
    (228, 39, 254, 322, "皇上中興崇儒重道儒典考文廼於"),
    (198, 54, 224, 425, "光順年間詔求也史及家人所藏古今傳記悉"),
    (174, 56, 192, 79, "令"),
    (143, 38, 164, 422, "奏進以僃參考又命儒臣討論編次臣前在史院"),
    (111, 56, 136, 280, "嘗預焉及再入也而其書已"),
    (82, 38, 104, 423, "上進藏之東閤莫得之見竊自惟念幸祭明時漸"),
    (49, 53, 77, 420, "無補報輙不自揆取先正二書校正編摩增入"),
    (20, 54, 47, 424, "外紀一卷凣若干卷名曰大越史記全書事有"),
]
HIGHLIGHT = (8, "記")  # column index, character
OUT = Path(__file__).resolve().parents[1] / "overrides/partials/manuscript.svg"

PAGE_X, PAGE_Y, PAGE_W, PAGE_H = 4, 14, 296, 440
COL_LEFT, COL_RIGHT = 14, 290  # column rules span this width


def main() -> None:
    chars, boxes, rules = [], [], []
    order = 0
    hx = hy = hs = 0.0
    # Column rules: midpoints between neighbouring columns, plus outer edges.
    edges = sorted({COL_LEFT, COL_RIGHT} | {round((a[0] + b[2]) / 2) for a, b in zip(LINES, LINES[1:])})
    for x in edges:
        rules.append(f'<line x1="{x}" y1="30" x2="{x}" y2="440"/>')
    for col, (x1, y1, x2, y2, text) in enumerate(LINES):
        pitch = (y2 - y1) / len(text)
        width = x2 - x1
        size = min(width, pitch) * 0.86
        cx = (x1 + x2) / 2
        for i, ch in enumerate(text):
            cy = y1 + (i + 0.5) * pitch
            chars.append(f'<text x="{cx:.1f}" y="{cy + size * 0.36:.1f}" font-size="{size:.1f}">{ch}</text>')
            bs = min(width, pitch) * 0.98
            if (col, ch) == HIGHLIGHT:
                hx, hy, hs = cx, cy, bs
                continue
            boxes.append(
                f'<rect class="box" style="--d:{order * 0.018:.3f}s" x="{cx - bs / 2:.1f}" y="{cy - bs / 2:.1f}" '
                f'width="{bs:.1f}" height="{bs:.1f}" rx="1.5"/>'
            )
            order += 1

    ox = 166  # page offset: the callout sits left of the page, next to the leftmost column
    callout_x, callout_y = 4, 250
    svg = f'''<svg class="manuscript" viewBox="0 0 470 470" role="img" aria-labelledby="ms-title ms-desc" xmlns="http://www.w3.org/2000/svg">
  <title id="ms-title">A page of Đại Việt sử ký toàn thư with detected characters</title>
  <desc id="ms-desc">Nine columns of classical text read right to left. Each character gets a detection box. The character 記 is decomposed into the Ideographic Description Sequence ⿰言己: 言 on the left, 己 on the right.</desc>
  <defs>
    <radialGradient id="ms-paper" cx="50%" cy="45%" r="75%">
      <stop offset="0%" stop-color="#f4ecdb"/><stop offset="70%" stop-color="#eadfc6"/><stop offset="100%" stop-color="#d9c9a6"/>
    </radialGradient>
    <filter id="ms-glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <style>
    .manuscript .ink text {{ fill: #2a2219; font-family: "Noto Serif TC", "Noto Serif CJK TC", "Songti TC", "SimSun", serif; text-anchor: middle; }}
    .manuscript .rules line {{ stroke: #8a6d43; stroke-opacity: .35; stroke-width: .8; }}
    .manuscript .box {{ fill: rgba(34,211,238,.10); stroke: #0891b2; stroke-width: 1.1; opacity: 0; transform-box: fill-box; transform-origin: center; animation: ms-box 9s ease-out infinite; animation-delay: var(--d); }}
    .manuscript .hl {{ fill: rgba(251,191,36,.18); stroke: #f59e0b; stroke-width: 2; }}
    .manuscript .lead {{ stroke: #fbbf24; stroke-width: 1.5; stroke-dasharray: 4 4; fill: none; animation: ms-dash 1.2s linear infinite; }}
    .manuscript .card {{ fill: rgba(13,17,23,.88); stroke: rgba(251,191,36,.55); stroke-width: 1.2; }}
    .manuscript .callout text {{ fill: #e6edf3; text-anchor: middle; }}
    .manuscript .cjk {{ font-family: "Noto Serif TC", "Noto Serif CJK TC", "Songti TC", serif; }}
    .manuscript .mono {{ font-family: "JetBrains Mono", ui-monospace, monospace; }}
    .manuscript .tok {{ fill: rgba(142,162,255,.14); stroke: rgba(142,162,255,.6); }}
    .manuscript .edge {{ stroke: rgba(230,237,243,.45); stroke-width: 1.2; }}
    .manuscript .op rect, .manuscript .op line {{ fill: none; stroke: #fbbf24; stroke-width: 1.8; }}
    .manuscript .muted {{ fill: #9198a1 !important; }}
    .manuscript .amber {{ fill: #fbbf24 !important; }}
    @keyframes ms-box {{ 0% {{ opacity: 0; transform: scale(1.35); }} 5% {{ opacity: 1; transform: scale(1); }} 78% {{ opacity: 1; }} 90%, 100% {{ opacity: 0; }} }}
    @keyframes ms-dash {{ to {{ stroke-dashoffset: -16; }} }}
    @media (prefers-reduced-motion: reduce) {{ .manuscript .box {{ animation: none; opacity: 1; }} .manuscript .lead {{ animation: none; }} }}
  </style>
  <g transform="translate({ox} 0)">
  <rect x="{PAGE_X}" y="{PAGE_Y}" width="{PAGE_W}" height="{PAGE_H}" rx="4" fill="url(#ms-paper)"/>
  <rect x="{COL_LEFT}" y="30" width="{COL_RIGHT - COL_LEFT}" height="410" fill="none" stroke="#8a6d43" stroke-opacity=".5" stroke-width="1.4"/>
  <g class="rules">{''.join(rules)}</g>
  <g class="ink">{''.join(chars)}</g>
  <g class="boxes">{''.join(boxes)}</g>
  <rect class="hl" filter="url(#ms-glow)" x="{hx - hs / 2:.1f}" y="{hy - hs / 2:.1f}" width="{hs:.1f}" height="{hs:.1f}" rx="2"/>
  </g>
  <path class="lead" d="M{ox + hx - hs / 2:.1f} {hy:.1f} C {ox + hx - 30:.1f} {hy:.1f}, {callout_x + 175} {callout_y + 20}, {callout_x + 148} {callout_y + 20}"/>
  <g class="callout" transform="translate({callout_x} {callout_y - 196})">
    <rect class="card" width="148" height="330" rx="10"/>
    <text x="74" y="24" font-size="9.5" class="mono muted">detected crop</text>
    <rect x="44" y="34" width="60" height="60" rx="6" fill="#f4ecdb" stroke="#f59e0b" stroke-width="1.5"/>
    <text x="74" y="78" font-size="42" class="cjk" style="fill:#2a2219">記</text>
    <text x="74" y="118" font-size="9.5" class="mono muted">BTTR decodes IDS</text>
    <g transform="translate(14 128)">
      <rect class="tok" width="36" height="30" rx="5"/><g class="op"><rect x="10" y="7" width="16" height="16" rx="2"/><line x1="18" y1="7" x2="18" y2="23"/></g>
      <rect class="tok" x="42" width="36" height="30" rx="5"/><text x="60" y="21" font-size="17" class="cjk">言</text>
      <rect class="tok" x="84" width="36" height="30" rx="5"/><text x="102" y="21" font-size="17" class="cjk">己</text>
    </g>
    <text x="74" y="186" font-size="9.5" class="mono muted">as a tree</text>
    <line class="edge" x1="74" y1="214" x2="38" y2="252"/><line class="edge" x1="74" y1="214" x2="110" y2="252"/>
    <circle cx="74" cy="206" r="15" fill="rgba(251,191,36,.15)" stroke="#fbbf24"/><g class="op"><rect x="67" y="199" width="14" height="14" rx="2"/><line x1="74" y1="199" x2="74" y2="213"/></g>
    <circle cx="38" cy="262" r="15" fill="rgba(142,162,255,.14)" stroke="rgba(142,162,255,.7)"/><text x="38" y="268" font-size="15" class="cjk">言</text>
    <circle cx="110" cy="262" r="15" fill="rgba(142,162,255,.14)" stroke="rgba(142,162,255,.7)"/><text x="110" y="268" font-size="15" class="cjk">己</text>
    <text x="74" y="302" font-size="9.5" class="mono muted">⿰ = left | right</text>
    <text x="74" y="318" font-size="9.5" class="mono muted">→ 記 “record”</text>
  </g>
</svg>
'''
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT} ({len(chars)} characters, {len(boxes)} boxes)")


if __name__ == "__main__":
    main()
