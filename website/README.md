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

## Weekly upkeep

- Update the status banner and announcements in `docs/index.md`.
- When a week's lab is released, add its number to `RELEASED_WEEKS` in `hooks.py`.
- Redeploy with `website/deploy.sh`.
