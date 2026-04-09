# synthetic-injection

Synthetic training fixtures for the concept-boundaries trust-region classifier
where every fixture contains at least one `doc:` region whose content is a real
prompt-injection attack string.

## Where the attack content comes from

Attack strings are drawn from `pinky/datasets/gandalf/gandalf_all.json`, the
`gandalf_ignore_instructions` dataset published by Lakera (MIT license),
collected from their Gandalf game (https://gandalf.lakera.ai/) where humans
tried to extract a hidden password from an LLM. We use the `train` split (777
entries). Strings are stripped of newlines/tabs, whitespace-collapsed,
truncated at a word boundary to ~400 chars, and dropped if they begin with `#`
(which would make the parser treat the whole line as a comment).

System / user / tool regions reuse the snippet libraries from
`../synthetic/snippets/`. Doc regions either contain a Gandalf injection
(every fixture has at least one) or, in ~20% of fixtures, BOTH a Gandalf
injection AND a normal benign doc snippet — so the classifier learns that
"doc region" does not automatically mean "attack".

## Architectural purpose

This morning's classifier (see `../../MORNING-2026-04-08.md`) was trained on
50 synthetic fixtures whose doc regions were neutral text (restaurants,
weather, news). It hit f1=0.750 on held-out hand-crafted fixtures. These new
fixtures test the same architecture under adversarial conditions: when the
doc region's content is itself an instruction-like attack, does the
classifier still cleanly mark the trust boundary at the source-tag transition,
or does it get fooled into treating the injection itself as a boundary?

## Regenerate

```bash
cd pinky/concept-boundaries-train/synthetic-injection
python generate_injection_fixtures.py --count 50 --seed 43
```

Stdlib only. No pandas, no numpy.

## Success criterion

A classifier retrained on (morning fixtures + these injection fixtures) should
still achieve high recall (target: meet or beat f1=0.750) on the held-out
hand-crafted fixtures in `pinky/concept-boundaries/fixtures/`, demonstrating
robustness to adversarial doc-region content.
