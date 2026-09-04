# Educational HTML page contract

Use this contract when the source can support the content. If a section is not supported by the source, keep the section and state that the source does not cover it. Do not invent material to fill the structure.

## Content structure

The document should contain these landmarks in a sensible reading order:

1. `header` with the lesson title, a one-sentence description grounded in the source, source names, and a visible scope note.
2. `main` with:
   - learning objectives
   - overview
   - key ideas, grouped into short sections
   - key terms and source-faithful definitions
   - worked examples or source figures/tables/formulas where they aid understanding
   - retrieval practice
   - a short quiz with feedback and answer explanations
   - takeaways
   - source notes and limits
3. `footer` with source attribution and the generated-page date if useful. Do not imply that the page is an official source document.

Every substantial factual section needs a nearby source marker. For a PDF, use markers such as `Source: p. 7` or `Sources: pp. 7-8`. For text input, use the file name and heading or section. Link markers to the source-notes section when that improves navigation.

## Teaching rules

- Prefer short paragraphs, meaningful subheadings, and one idea per block.
- State the source's claim first, then explain it in simpler language. Preserve the source's conditions and exceptions.
- Use bold sparingly for terms the student should notice. Use code, math, and blockquotes only when the source uses them or they improve fidelity.
- Treat examples as evidence only when the source provides them. Label an original analogy or hypothetical example as such and keep it separate from the source explanation.
- Write questions with enough context to stand alone. Put the answer and explanation in a disclosure element or an equivalent accessible control.
- For multiple-choice questions, include one correct answer, plausible distractors based on common misunderstandings in the source, and a brief explanation for every option when practical.

## Interface requirements

- Use a responsive single-column reading layout that can expand to a two-column layout with navigation on wide screens.
- Include a skip link, landmark elements, ordered heading levels, descriptive link text, and visible `:focus-visible` styles.
- Keep body text comfortable to read and keep long lines constrained. Do not make the student hunt for the next action.
- Use `<details>` and `<summary>` or equivalent accessible controls for answer reveals. Do not hide required lesson content behind JavaScript.
- If a quiz is auto-scored, show the score only after submission, identify unanswered items, and let the student try again without losing the explanations.
- Respect `prefers-reduced-motion` and do not make progress animations necessary for comprehension.

## Self-contained Tailwind output

Use Tailwind utilities in the markup and generate only the classes the page needs. Inline the resulting CSS in a `<style>` element. If a Tailwind compiler is unavailable, use a local, minimal compiled utility stylesheet that matches the used Tailwind utilities and keep the markup utility-based. Never add a runtime CDN script to an offline deliverable.
