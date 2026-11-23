"""Every relative link in README.md and docs/ points to an existing file, and every #anchor to a heading of it
(GitHub's anchor rules)."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _anchor(heading):
    a = heading.strip().lower()
    a = re.sub(r"[^\w\- ]", "", a)
    return a.replace(" ", "-")


def _anchors(path):
    out = set()
    in_code = False
    for line in open(path, encoding="utf-8"):
        if line.startswith("```"):
            in_code = not in_code
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m and not in_code:
            out.add(_anchor(m.group(2)))
    return out


def _md_files():
    yield os.path.join(ROOT, "README.md")
    for dp, _, fn in os.walk(os.path.join(ROOT, "docs")):
        for f in fn:
            if f.endswith(".md"):
                yield os.path.join(dp, f)
    yield os.path.join(ROOT, "examples", "README.md")


def test_links_resolve():
    bad = []
    for md in _md_files():
        text = re.sub(r"```.*?```", "", open(md, encoding="utf-8").read(), flags=re.S)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path, _, frag = target.partition("#")
            dest = os.path.normpath(os.path.join(os.path.dirname(md), path)) if path else md
            if not os.path.exists(dest):
                bad.append(f"{os.path.relpath(md, ROOT)}: {target} (no such file)")
            elif frag and dest.endswith(".md") and frag not in _anchors(dest):
                bad.append(f"{os.path.relpath(md, ROOT)}: {target} (no such heading)")
    assert not bad, "\n".join(bad)
