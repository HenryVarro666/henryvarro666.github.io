"""Build the password-protected Vault section into static/vault/ and static/zh/vault/.

The Vault is an ordinary Hugo section whose Markdown sources are kept OUTSIDE this repository.
This script renders them with the site's own templates in a temporary copy of the repository,
encrypts every rendered page with StatiCrypt, and writes only the encrypted pages back here.

    STATICRYPT_PASSWORD='...' python3 tools/vault/build.py /path/to/vault-src

The source folder is laid out like a section under content/:

    vault-src/
      _index.md              title, intro, the "Vault" menu entry, and the cascade (toc, noindex)
      _index.zh.md           the same cascade for Chinese pages
      some-post/index.md     one post per folder; add index.zh.md for a Chinese version

Needs Hugo (the version in .github/workflows/hugo.yml) and Node.js; StatiCrypt is fetched with
npx. The password is read from the environment and is never written to disk. The salt in
staticrypt.json is not secret; keeping it stable lets a remembered login survive a rebuild.
Files other than HTML pages (images, downloads) are not encrypted, so they are not copied.
"""
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
STATICRYPT = "staticrypt@3.5.4"
TARGETS = ("vault", "zh/vault")  # rendered trees, relative to both the build output and static/


def run(cmd, **kwargs):
    result = subprocess.run(cmd, text=True, capture_output=True, **kwargs)
    if result.returncode:
        sys.exit(f"{cmd[0]} failed:\n{result.stdout}{result.stderr}")
    return result


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1]).resolve()
    if not (src / "_index.md").is_file():
        sys.exit(f"{src} has no _index.md")
    if ROOT in src.parents or src == ROOT:
        sys.exit("Keep the Vault sources outside the repository, so the plaintext is never committed.")
    password = os.environ.get("STATICRYPT_PASSWORD")
    if not password:
        sys.exit("Set STATICRYPT_PASSWORD in the environment.")

    with tempfile.TemporaryDirectory() as tmp:
        site, out, enc = Path(tmp, "site"), Path(tmp, "out"), Path(tmp, "enc")
        shutil.copytree(ROOT, site, ignore=shutil.ignore_patterns(".git", "public", "resources", "preview"))
        for target in TARGETS:  # the encrypted pages would collide with the pages rendered below
            shutil.rmtree(site / "static" / target, ignore_errors=True)
        shutil.copytree(src, site / "content" / "vault")

        env = dict(os.environ, HUGO_PARAMS_INLINECSS="true", HUGO_ENVIRONMENT="production")
        env.pop("STATICRYPT_PASSWORD")
        run(["hugo", "--gc", "--minify", "-d", str(out)], cwd=site, env=env)

        pages = [p for target in TARGETS for p in sorted((out / target).rglob("index.html"))]
        if not pages:
            sys.exit("Hugo rendered no Vault pages.")
        skipped = [p.relative_to(out) for target in TARGETS for p in sorted((out / target).rglob("*"))
                   if p.is_file() and p.name != "index.html"]

        written = []
        for page in pages:
            rel = page.relative_to(out)
            plain = page.read_text(encoding="utf-8")
            if "noindex" not in plain:
                sys.exit(f"{rel} is missing the noindex robots tag; check the cascade in _index.md.")
            # StatiCrypt resolves its paths against the working directory, so run it from this folder.
            run(["npx", "--yes", STATICRYPT, os.path.relpath(page, HERE),
                 "-d", os.path.relpath(enc / rel.parent, HERE),
                 "-c", "staticrypt.json", "-t", "password_template.html",
                 "--short", "--remember", "30",
                 "--template-title", "Private page",
                 "--template-instructions", "This page is password-protected.",
                 "--template-button", "Open",
                 "--template-remember", "Stay signed in on this device",
                 "--template-color-primary", "#3f4a54",
                 "--template-color-secondary", "#eceae6"],
                cwd=HERE, env=dict(os.environ, STATICRYPT_PASSWORD=password))
            cipher = (enc / rel).read_text(encoding="utf-8")
            title = re.search(r"<title>(.*?)</title>", plain, re.S)
            heading = re.search(r"<h1[^>]*>(.*?)</h1>", plain, re.S)
            for probe in (title, heading):
                text = html.unescape(re.sub(r"<[^>]+>", "", probe.group(1))).strip() if probe else ""
                if text and text != "Private page" and text in cipher:
                    sys.exit(f"{rel}: plaintext found in the encrypted page; nothing was written.")
            written.append((rel, len(cipher)))

        for target in TARGETS:
            shutil.rmtree(ROOT / "static" / target, ignore_errors=True)
            if (enc / target).is_dir():
                shutil.copytree(enc / target, ROOT / "static" / target)

    for rel, size in written:
        print(f"encrypted  static/{rel}  ({size:,} bytes)")
    for rel in skipped:
        print(f"not copied (not an HTML page, so it cannot be encrypted): {rel}")


if __name__ == "__main__":
    main()
