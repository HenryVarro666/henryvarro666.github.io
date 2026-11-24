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
Only HTML pages are encrypted. An image that a post embeds from its own folder is written into
the page as a data URI, so it is encrypted with the page. Any other file (a download, say) is
not copied, and the script warns if a page links to one.
"""
import base64
import html
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
STATICRYPT = "staticrypt@3.5.4"
TARGETS = ("vault", "zh/vault")  # rendered trees, relative to both the build output and static/


def run(cmd, **kwargs):
    result = subprocess.run(cmd, text=True, capture_output=True, **kwargs)
    if result.returncode:
        sys.exit(f"{cmd[0]} failed:\n{result.stdout}{result.stderr}")
    return result


def inline_images(plain, out, inlined):
    """Replace <img src> that points at a file rendered under the Vault with a data URI."""
    def swap(match):
        path = (out / unquote(match.group(2)).lstrip("/")).resolve()
        if not path.is_file() or not any((out / t).resolve() in path.parents for t in TARGETS):
            return match.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        inlined.add(path)
        return f'{match.group(1)}"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"'
    # the minifier may drop the quotes around an attribute value
    return re.sub(r'(<img\b[^>]*?\bsrc=)["\']?(/[^"\'\s>]+)["\']?', swap, plain)


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
        others = [p for target in TARGETS for p in sorted((out / target).rglob("*"))
                  if p.is_file() and p.name != "index.html"]

        written, inlined, linked = [], set(), set()
        for page in pages:
            rel = page.relative_to(out)
            plain = page.read_text(encoding="utf-8")
            if "noindex" not in plain:
                sys.exit(f"{rel} is missing the noindex robots tag; check the cascade in _index.md.")
            plain = inline_images(plain, out, inlined)
            page.write_text(plain, encoding="utf-8")
            for other in others:  # a link written as an absolute path, or relative to the page
                url = "/" + other.relative_to(out).as_posix()
                local = r'(?:href|src)=["\']?(?:\./)?' + re.escape(other.name) + r'(?=["\'\s>#?])'
                if url in plain or (other.parent == page.parent and re.search(local, plain)):
                    linked.add(other)
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
    for path in others:
        rel = path.relative_to(out)
        if path.resolve() in inlined:
            print(f"inlined    {rel}")
        elif path in linked:
            print(f"WARNING    a Vault page links to {rel}, which is not published; the link will be broken")
        else:
            print(f"not copied (not an HTML page, so it cannot be encrypted): {rel}")


if __name__ == "__main__":
    main()
