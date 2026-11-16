# Course site

Static documentation site for the course, built with MkDocs Material and deployed to Vercel
as plain files. The lab READMEs remain the single source of truth: `hooks.py` copies them
into `docs/weeks/` and `docs/guides/` at build time and rewrites their links. Edit the labs,
not the generated pages.

Hand-written pages: `docs/index.md` (home, status banner, announcements),
`getting-started.md`, `syllabus.md`, `assessments.md`, `resources.md`, `help.md`.

## Preview locally

```bash
uv run --project website mkdocs serve -f website/mkdocs.yml
```

## Deploy

```bash
cp website/.env.example website/.env   # add the classroom account's VERCEL_TOKEN
website/deploy.sh                      # production; use --preview for a preview URL
```

The script creates the Vercel project through the API with no Git repository, uploads from a
temporary folder outside this checkout, and runs the Vercel CLI with a throwaway config, so it
never touches a GitHub connection or an existing `vercel login`. It refuses to deploy if the
project has a Git connection.

Links to notebooks and scripts point to GitHub at `DOCS_SOURCE_REF` (default `main`). Until the
merged labs are on `main`, set it to the branch that holds them.

## Where things live

| To change | Edit |
| --- | --- |
| Announcements | `docs/index.md` (the landing page renders its body) |
| Current week, status labels, midterm countdown | `extra:` in `mkdocs.yml` (`current_week`, `current_label`, `midterm_date`) |
| Week card titles and one-line blurbs | `WEEKS` in `hooks.py` |
| Landing page layout (hero, status strip, announcements, pipeline, schedule) | `overrides/home.html` |
| Colours, fonts, GitHub-style docs look | `docs/stylesheets/extra.css` (phase colours are the `--ph-*` tokens) |
| Hero illustration | `tools/make_manuscript_svg.py`, then rerun it to regenerate `overrides/partials/manuscript.svg` |

The hero illustration uses a real NomNaOCR page (the preface of *Đại Việt sử ký toàn thư*,
`DVSKTT_thu_III_2a`) with its actual column layout and transcription.

## Weekly upkeep

- Move `current_week` forward in `mkdocs.yml` and add an announcement in `docs/index.md`.
- When a week's lab is released, add its number to `RELEASED_WEEKS` in `hooks.py`.
- Update the status band text in `overrides/home.html` when the break ends.
- Redeploy with `website/deploy.sh`.
