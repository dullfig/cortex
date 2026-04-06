# pinky

Experimental sidekick to cortex. Named after Pinky from *Pinky and the Brain* —
this is where ideas get tried before the Brain (the production crates) takes
them seriously.

## Rules

- **Not in the workspace.** Each experiment is its own standalone Cargo project
  that path-deps on `../../cortex`. The root `Cargo.toml` excludes `pinky/` from
  the workspace, so `cargo build --workspace` and CI never touch this directory.
- **Looser standards apply.** Throwaway scripts, half-finished prototypes, weird
  deps, broken builds — all fine. Pinky exists so the main crates can stay
  clean.
- **Graduation is `git mv`.** When an experiment proves something and earns a
  place in production, move it up one level into a real workspace member, add
  it to `members` in the root `Cargo.toml`, and clean it up.
- **Each experiment has a README** explaining what it's testing, what it found,
  and what cortex changes (if any) it depends on.

## Current experiments

| Directory | Status | What it tests |
|-----------|--------|---------------|
| [`concept-boundaries/`](concept-boundaries/) | scaffold | Per-token provenance + (eventually) attention-discovered concept boundaries on mixed-trust input |
