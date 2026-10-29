"""Turn RECORDING_SCRIPT.md into a clean, readable PDF (RECORDING_SCRIPT.pdf).

Needs the `markdown` package and Node + Playwright. Run from anywhere.
"""
import os
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
html_body = markdown.markdown((ROOT / "RECORDING_SCRIPT.md").read_text(), extensions=["sane_lists"])

# keep each take together on a page: wrap every section that starts at an <h2>
parts = re.split(r"(?=<h2>)", html_body)
html_body = parts[0] + "".join(f"<section>{p}</section>" if p.startswith("<h2>Take") else p for p in parts[1:])
html_body = html_body.replace("<code>(beat)</code>", '<span class="beat">beat</span>')

css = """
@font-face{font-family:M;font-weight:500;src:url(../assets/fonts/montserrat-latin-500-normal.woff2)}
@font-face{font-family:M;font-weight:600;src:url(../assets/fonts/montserrat-latin-600-normal.woff2)}
@font-face{font-family:M;font-weight:800;src:url(../assets/fonts/montserrat-latin-800-normal.woff2)}
@page{size:Letter;margin:.7in .75in}
*{box-sizing:border-box}
body{font:500 12.5pt/1.6 M,Inter,sans-serif;color:#232220;margin:0}
h1{font-weight:800;font-size:25pt;line-height:1.15;margin:0 0 6pt;letter-spacing:-.02em}
h1+p{color:#6B655E;margin:0 0 4pt}
h2{font-weight:800;font-size:14pt;margin:14pt 0 6pt;break-after:avoid;padding-left:12pt;border-left:5pt solid #F26A1B;line-height:1.25}
section{break-inside:avoid;margin:0 0 16pt}
p{margin:0 0 9pt}
em{color:#6B655E;font-size:10.5pt;font-style:italic;display:block;background:#FAF8F5;padding:7pt 10pt;border-radius:7pt}
strong{color:#E5570C;font-weight:800}
.beat{display:inline-block;font-size:8.5pt;font-weight:600;color:#9A9288;border:1pt solid #D9D1C6;border-radius:99pt;padding:0 6pt;vertical-align:1pt}
hr{border:0;border-top:1pt solid #ECE5DB;margin:12pt 0}
ul,ol{padding-left:18pt;margin:0 0 9pt}li{margin-bottom:5pt}
blockquote{margin:10pt 0;padding:8pt 12pt;background:#FFF3E3;border-left:5pt solid #F2A71B;border-radius:6pt}
blockquote p{margin:0}
code{font-size:10pt;background:#F3EEE7;padding:0 3pt;border-radius:3pt}
"""
out_html = ROOT / "build" / "recording_script.html"
out_html.write_text(f'<!doctype html><meta charset="utf-8"><style>{css}</style><body>{html_body}</body>')

js = """
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://%s'); await p.evaluate(() => document.fonts.ready);
  await p.pdf({ path: '%s', format: 'Letter', printBackground: True, preferCSSPageSize: true });
  await b.close();
})();
""".replace("True", "true") % (out_html, ROOT / "RECORDING_SCRIPT.pdf")
node_path = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
subprocess.run(["node", "-e", js], check=True, env={**os.environ, "NODE_PATH": node_path})
print("wrote", ROOT / "RECORDING_SCRIPT.pdf")
