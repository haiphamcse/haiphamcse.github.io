# Lectures (edtrace viewer)

Interactive edtrace traces, served at `/blog/lectures/`.

- **URL:** `https://haiphamcse.github.io/blog/lectures/?trace=<name>&step=<n>`
  e.g. `?trace=mllm_finetuning&step=0`
- Bare `/blog/lectures/` shows an input box to type a trace name.

## Contents
- `index.html`, `assets/` — the built React viewer (from `edtrace/frontend`).
- `var/traces/*.json` — the trace files (one per lecture).
- `images/` — images/plots referenced by traces.

## Add / update a trace
From `C:\Personal\Working\CS336\notes`:
```bat
publish_trace.bat <name>        REM regenerates notes\<name>.py and copies JSON here
```
Then commit + push this repo. GitHub Pages serves it statically.

## Rebuild the viewer (rare)
Only when the edtrace frontend code changes:
```bat
build_lectures.bat              REM in C:\Personal\Working\CS336\notes
```
It builds with base `/blog/lectures/` and refreshes `index.html` + `assets/` here
(leaving `var/traces/` and `images/` untouched).

## Local preview
```bat
cd C:\Personal\Working\VinAI\PersonalWeb\haiphamcse.github.io
python -m http.server 8099
REM open http://localhost:8099/blog/lectures/?trace=mllm_finetuning&step=0
```
