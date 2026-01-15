import html
import os
import re


MD_FILE = "Presentation_Materials.md"
HTML_FILE = "Presentation_Enhanced.html"

MAX_BLOCKS_PER_SLIDE = 6
MAX_BULLETS_PER_SLIDE = 5


def inline_format(text):
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def parse_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    title = None
    cover_blocks = []
    sections = []
    current = None
    in_code = False
    code_lang = ""
    code_lines = []

    def flush_code():
        nonlocal code_lines, code_lang, current, cover_blocks
        if not code_lines:
            return
        block = {
            "type": "code",
            "lang": code_lang.strip(),
            "text": "\n".join(code_lines),
        }
        if current is None:
            cover_blocks.append(block)
        else:
            current["blocks"].append(block)
        code_lines = []
        code_lang = ""

    def add_block(block):
        if current is None:
            cover_blocks.append(block)
        else:
            current["blocks"].append(block)

    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith("```"):
            if in_code:
                in_code = False
                flush_code()
            else:
                in_code = True
                code_lang = line.replace("```", "")
            continue
        if in_code:
            code_lines.append(line)
            continue

        stripped = line.strip()
        if not stripped:
            add_block({"type": "blank"})
            continue
        if stripped.startswith("# "):
            if title is None:
                title = stripped[2:].strip()
            else:
                add_block({"type": "h1", "text": stripped[2:].strip()})
            continue
        if stripped.startswith("## "):
            if current:
                sections.append(current)
            current = {"title": stripped[3:].strip(), "blocks": []}
            continue
        if stripped.startswith("### "):
            add_block({"type": "h3", "text": stripped[4:].strip()})
            continue
        if stripped.startswith("> "):
            add_block({"type": "quote", "text": stripped[2:].strip()})
            continue
        if re.match(r"^\d+\.\s+", stripped):
            add_block({"type": "ol", "text": stripped})
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            add_block({"type": "ul", "text": stripped})
            continue
        add_block({"type": "p", "text": stripped})

    if current:
        sections.append(current)

    return title, cover_blocks, sections


def build_lists(blocks):
    rendered = []
    ul_items = []
    ol_items = []

    def flush_lists():
        nonlocal ul_items, ol_items
        if ul_items:
            rendered.append(
                "<ul>" + "".join(f"<li>{inline_format(item)}</li>" for item in ul_items) + "</ul>"
            )
            ul_items = []
        if ol_items:
            rendered.append(
                "<ol>" + "".join(f"<li>{inline_format(item)}</li>" for item in ol_items) + "</ol>"
            )
            ol_items = []

    for block in blocks:
        btype = block["type"]
        if btype == "ul":
            ul_items.append(block["text"][2:].strip())
            continue
        if btype == "ol":
            item_text = re.sub(r"^\d+\.\s+", "", block["text"]).strip()
            ol_items.append(item_text)
            continue
        flush_lists()
        if btype == "h3":
            rendered.append(f"<h3>{inline_format(block['text'])}</h3>")
        elif btype == "h1":
            rendered.append(f"<h1>{inline_format(block['text'])}</h1>")
        elif btype == "quote":
            rendered.append(f"<blockquote>{inline_format(block['text'])}</blockquote>")
        elif btype == "code":
            lang_class = f" language-{block['lang']}" if block["lang"] else ""
            rendered.append(
                f"<pre><code class=\"{lang_class.strip()}\">{html.escape(block['text'])}</code></pre>"
            )
        elif btype == "p":
            rendered.append(f"<p>{inline_format(block['text'])}</p>")
        elif btype == "blank":
            rendered.append("<div class=\"spacer\"></div>")

    flush_lists()
    return "\n".join(rendered)


def split_blocks_into_slides(section_title, blocks):
    slides = []
    current = {"title": section_title, "subtitle": None, "blocks": []}
    bullet_count = 0

    def flush():
        nonlocal current, bullet_count
        if current["blocks"] or current["subtitle"]:
            slides.append(current)
        current = {"title": section_title, "subtitle": None, "blocks": []}
        bullet_count = 0

    for block in blocks:
        if block["type"] == "h3":
            flush()
            current["subtitle"] = block["text"]
            continue

        if block["type"] in {"code", "quote"} and current["blocks"]:
            flush()

        if block["type"] in {"ul", "ol"}:
            bullet_count += 1
        if len(current["blocks"]) >= MAX_BLOCKS_PER_SLIDE or bullet_count > MAX_BULLETS_PER_SLIDE:
            flush()

        current["blocks"].append(block)

        if block["type"] == "code":
            flush()

    flush()
    return slides


def generate_html(title, cover_blocks, sections, output_path):
    cover_html = build_lists(cover_blocks)
    slide_sections = []
    slide_index = 1
    for section in sections:
        section_slides = split_blocks_into_slides(section["title"], section["blocks"])
        for slide in section_slides:
            body_html = build_lists(slide["blocks"])
            subtitle_html = (
                f"<div class=\"subtitle\">{inline_format(slide['subtitle'])}</div>"
                if slide["subtitle"]
                else ""
            )
            slide_sections.append(
                f"""
                <section class="slide">
                  <header>
                    <div class="meta">
                      <span class="kicker">SLIDE {slide_index:02d}</span>
                      <h2>{inline_format(slide['title'])}</h2>
                      {subtitle_html}
                    </div>
                    <div class="bar"></div>
                  </header>
                  <div class="content">
                    {body_html}
                  </div>
                </section>
                """
            )
            slide_index += 1

    html_doc = f"""<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(title or "Presentation")}</title>
    <style>
      :root {{
        --bg: #f2f4f8;
        --ink: #0f1a2b;
        --accent: #ff7f3f;
        --accent-2: #1a4a8b;
        --paper: #ffffff;
        --shadow: rgba(15, 26, 43, 0.16);
        --muted: #5a6372;
      }}
      * {{
        box-sizing: border-box;
      }}
      body {{
        margin: 0;
        font-family: "IBM Plex Sans KR", "Spoqa Han Sans Neo", "Pretendard", sans-serif;
        background: radial-gradient(circle at 15% 15%, #ffffff, #eef2f6 60%, #e7ecf3);
        color: var(--ink);
      }}
      .deck {{
        display: flex;
        flex-direction: column;
        gap: 48px;
        padding: 48px 32px 72px;
      }}
      .slide {{
        background: var(--paper);
        border-radius: 28px;
        box-shadow: 0 22px 50px var(--shadow);
        padding: 48px 56px;
        min-height: 70vh;
        aspect-ratio: 16 / 9;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        position: relative;
        overflow: hidden;
        animation: floatIn 0.6s ease both;
      }}
      .cover {{
        background: linear-gradient(135deg, #0f213a 0%, #193458 55%, #132a45 100%);
        color: #fef6ee;
        position: relative;
        overflow: hidden;
      }}
      .cover::after {{
        content: "";
        position: absolute;
        width: 420px;
        height: 420px;
        border-radius: 50%;
        background: rgba(233, 115, 63, 0.18);
        right: -140px;
        top: -120px;
      }}
      .cover h1 {{
        font-size: clamp(2.6rem, 4vw, 4rem);
        margin: 0 0 18px;
        letter-spacing: 0.02em;
      }}
      .cover blockquote {{
        border-left: 4px solid var(--accent);
        padding-left: 16px;
        margin: 12px 0;
        font-size: 1.05rem;
        color: #f8ead8;
      }}
      header {{
        margin-bottom: 24px;
      }}
      .meta {{
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}
      .kicker {{
        font-size: 0.85rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--muted);
      }}
      header h2 {{
        font-size: clamp(2rem, 3vw, 3rem);
        margin: 0 0 12px;
        color: var(--accent-2);
      }}
      .subtitle {{
        font-size: 1.1rem;
        color: var(--muted);
        font-weight: 600;
      }}
      .bar {{
        height: 6px;
        width: 120px;
        background: var(--accent);
        border-radius: 999px;
      }}
      h3 {{
        margin: 22px 0 8px;
        font-size: 1.4rem;
        color: var(--accent-2);
      }}
      p, li {{
        font-size: 1.1rem;
        line-height: 1.65;
      }}
      ul, ol {{
        margin: 10px 0 18px 22px;
      }}
      blockquote {{
        margin: 18px 0;
        padding: 12px 18px;
        background: #f6efe7;
        border-left: 4px solid var(--accent);
        font-size: 1.05rem;
      }}
      code {{
        background: #f2e6d7;
        padding: 2px 6px;
        border-radius: 6px;
        font-family: "Fira Code", "Nanum Gothic Coding", monospace;
        font-size: 0.95rem;
      }}
      pre {{
        background: #101c2a;
        color: #f8efe7;
        padding: 18px;
        border-radius: 14px;
        overflow-x: auto;
        font-size: 0.95rem;
      }}
      .spacer {{
        height: 10px;
      }}
      @keyframes floatIn {{
        from {{
          opacity: 0;
          transform: translateY(12px);
        }}
        to {{
          opacity: 1;
          transform: translateY(0);
        }}
      }}
      @media (max-width: 900px) {{
        .slide {{
          padding: 36px 28px;
          min-height: auto;
        }}
        .deck {{
          padding: 32px 18px 48px;
        }}
      }}
      @media print {{
        body {{
          background: #ffffff;
        }}
        .deck {{
          gap: 0;
          padding: 0;
        }}
        .slide {{
          page-break-after: always;
          box-shadow: none;
          border-radius: 0;
          min-height: auto;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="deck">
      <section class="slide cover">
        <h1>{inline_format(title or "Presentation")}</h1>
        {cover_html}
      </section>
      {"".join(slide_sections)}
    </div>
  </body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(current_dir, MD_FILE)
    html_path = os.path.join(current_dir, HTML_FILE)

    title, cover_blocks, sections = parse_markdown(md_path)
    generate_html(title, cover_blocks, sections, html_path)
    print(f"HTML presentation saved to {html_path}")
