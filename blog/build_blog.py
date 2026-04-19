#!/usr/bin/env python3
"""
Build static blog HTML from blog/src/*.md (YAML front matter).

Usage (from repository root):
  python blog/build_blog.py

Requires: pip install -r blog/requirements-blog.txt
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import markdown
import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

EXTENSIONS = [
    "markdown.extensions.tables",
    "pymdownx.superfences",
    "pymdownx.highlight",
    "pymdownx.arithmatex",
]

EXTENSION_CONFIGS = {
    "pymdownx.superfences": {},
    "pymdownx.highlight": {
        "use_pygments": True,
        "pygments_style": "friendly",
    },
    "pymdownx.arithmatex": {
        "generic": True,
    },
}

try:
    import pygments  # noqa: F401
except ImportError:
    EXTENSION_CONFIGS["pymdownx.highlight"]["use_pygments"] = False

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title_escaped} — Hai Pham</title>
  <meta name="description" content="{desc_escaped}">
  <link rel="icon" type="image/png" href="../images/icon.jpg">
  <link rel="stylesheet" href="../stylesheet.css">
  <link rel="stylesheet" href="blog.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">
</head>
<body>
  <div class="container blog-wrap">
    <nav class="blog-nav"><a href="../index.html">← Home</a> · <a href="index.html">Blog</a></nav>
    <article class="blog-post">
      <h1 class="blog-title">{title_html}</h1>
      <p class="blog-meta">{date_display}</p>
___CONTENT___
    </article>
  </div>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous"></script>
  <script>
  document.addEventListener("DOMContentLoaded", function() {{
    var el = document.querySelector(".blog-post");
    if (!el || typeof renderMathInElement === "undefined") return;
    renderMathInElement(el, {{
      delimiters: [
        {{left: "$$", right: "$$", display: true}},
        {{left: "$", right: "$", display: false}},
        {{left: "\\\\(", right: "\\\\)", display: false}},
        {{left: "\\\\[", right: "\\\\]", display: true}}
      ],
      strict: false,
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
    }});
  }});
  </script>
</body>
</html>
"""


def split_front_matter(text: str) -> tuple[dict, str]:
    text = text.lstrip("\ufeff")
    if not text.startswith("---"):
        raise ValueError("File must start with YAML front matter (---)")
    end = text.find("\n---", 3)
    if end == -1:
        raise ValueError("Missing closing --- for front matter")
    fm_raw = text[3:end]
    body = text[end + 4 :].lstrip("\n")
    meta = yaml.safe_load(fm_raw) or {}
    if not isinstance(meta, dict):
        raise ValueError("Front matter must be a mapping")
    return meta, body


def md_to_html(md: str) -> str:
    return markdown.markdown(
        md,
        extensions=EXTENSIONS,
        extension_configs=EXTENSION_CONFIGS,
    )


def format_date(meta: dict) -> str:
    d = meta.get("date")
    if d is None:
        return ""
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)


def main() -> int:
    if not SRC.is_dir():
        print("Missing blog/src directory", file=sys.stderr)
        return 1

    md_files = sorted(SRC.glob("*.md"))
    if not md_files:
        print("No .md files in blog/src", file=sys.stderr)
        return 1

    for path in md_files:
        raw = path.read_text(encoding="utf-8")
        try:
            meta, body = split_front_matter(raw)
        except ValueError as e:
            print(f"{path.name}: {e}", file=sys.stderr)
            return 1

        slug = meta.get("slug")
        title = meta.get("title")
        if not slug or not title:
            print(f"{path.name}: `slug` and `title` are required in front matter", file=sys.stderr)
            return 1

        safe_slug = str(slug).strip()
        if not safe_slug or ".." in safe_slug or "/" in safe_slug:
            print(f"{path.name}: invalid slug", file=sys.stderr)
            return 1

        desc = meta.get("description") or title
        content_html = md_to_html(body)
        indented = "\n".join("      " + line if line.strip() else line for line in content_html.splitlines())

        title_escaped = html.escape(str(title), quote=True)
        desc_escaped = html.escape(str(desc), quote=True)
        title_html = html.escape(str(title))
        date_display = html.escape(format_date(meta))

        out_html = TEMPLATE.format(
            title_escaped=title_escaped,
            desc_escaped=desc_escaped,
            title_html=title_html,
            date_display=date_display,
        )
        out_html = out_html.replace("___CONTENT___", indented)

        out_path = ROOT / f"{safe_slug}.html"
        out_path.write_text(out_html, encoding="utf-8")
        print(f"Wrote {out_path.relative_to(ROOT.parent)}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
