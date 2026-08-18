# Regenerating the example documentation pages

The pages under `docs/*.md` for the six examples
(`burgers_eq.md`, `heat_eq.md`, `cavity_flow_steady.md`,
`cylinder_flow_steady.md`, `BFS_flow_steady.md`, `channel_flow_transient.md`)
are **hand-authored**. The corresponding notebooks in `examples/` are the
source of truth.

This is a deliberate choice: raw `nbconvert` output leaks machine-specific
paths, training printouts, and runtime warnings into the docs (see the
documentation cleanup plan). A hand-authored page keeps the code faithful to
the notebook while controlling the narrative.

## When to update a page

Update a page whenever its notebook changes materially — new API usage, new
sections, changed hyperparameters, or when the reported loss numbers no
longer reflect a current run. When you do:

1. Re-run the notebook against current `src/` (or at minimum audit every
   `df.*` call statically against `src/deepflow/`).
2. Update the code blocks to match the notebook cell-for-cell (simplified
   where prose explains the skipped parts).
3. Re-export the figures the page shows:

   ```bash
   jupyter nbconvert --execute --to notebook examples/<case>/<case>.ipynb
   ```

   and copy the plots you want into `docs/static/examples/<case>/` with the
   semantic names used by the page (e.g. `velocity_field.png`,
   `loss_curve.png`). Keep the names stable so links don't break.
4. Update the reference-run numbers in the "At a glance" / result note, and
   bump the **Verified against** line to the current version and commit:
   `DeepFlow v0.1.3 (commit <sha>)`.
5. `mkdocs build --strict` must pass (CI enforces this on docs PRs).

## Asset layout

- `docs/static/examples/<case>/` — figures used by the example pages.
- `static/` (repo root) — figures used by the GitHub README.
- `docs/static/` — figures used by the rest of the site (Quick Start etc.).

There is no separate `docs/img/` tree anymore.
