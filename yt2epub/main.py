# ────────── GLOBAL DEFAULTS ────────── #
DEFAULT_TITLE      = "Untitled"
DEFAULT_AUTHOR     = "Anonymous"
DEFAULT_PUBLISHER  = "Esadev"
DEFAULT_LANGUAGE   = "en"
DEFAULT_OUT_NAME   = "my_book"
COVER_H, COVER_W   = 768, 512          # 3:2 vertical, tiny thumbnail size
DEFAULT_TEXT_COLOR = "dimgray"     
DEFAULT_STROKE_COLOR = "white"    
# ────────────────────────────────────── #

import threading
import subprocess
import os, re, uuid, unicodedata, tkinter as tk, tempfile, random
import yt_dlp
import google.generativeai as genai
import torch
import time
import shutil
from tkinter import messagebox
from tkinterdnd2 import TkinterDnD
from ebooklib import epub
from PIL import Image, ImageTk
from PIL import ImageDraw, ImageFont
from faster_whisper import WhisperModel
from pathlib import Path
from diffusers import StableDiffusionXLPipeline
from diffusers import UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import ImageOps
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np
import webbrowser

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

# Load once & cache
_sdxl_lightning_pipe = None

# ─── helpers ─────────────────────────────────────────────────────── #
HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)$")
TMP_DIR   = tempfile.gettempdir()
COVER_FN  = os.path.join(TMP_DIR, "cover.png")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_cover_image(prompt: str, save_path: str = "cover.png", steps: int = 4) -> torch.Tensor:
    global _sdxl_lightning_pipe

    if _sdxl_lightning_pipe is None:
        print("[SDXL Lightning] Initializing pipeline...")

        base_model  = "stabilityai/stable-diffusion-xl-base-1.0"
        ckpt_repo   = "ByteDance/SDXL-Lightning"
        ckpt_file   = "sdxl_lightning_4step_unet.safetensors"
        cache_dir   = "D:/AI/models"  

        # Load and override UNet
        config = UNet2DConditionModel.load_config(
            pretrained_model_name_or_path=base_model,
            subfolder="unet",
            cache_dir=cache_dir
        )

        unet = UNet2DConditionModel.from_config(config).to("cuda", torch.float16)

        unet.load_state_dict(load_file(
            hf_hub_download(repo_id=ckpt_repo, filename=ckpt_file, cache_dir=cache_dir),
            device="cuda"
        ))

        # Build the full pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir=cache_dir
        ).to("cuda")

        # Set scheduler to Lightning-compatible mode
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing"
        )

        _sdxl_lightning_pipe = pipe

    print(f"[SDXL Lightning] Generating image for: {prompt}")
    image = _sdxl_lightning_pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0).images[0]
    image = ImageOps.fit(image, (COVER_W, COVER_H), method=Image.LANCZOS, centering=(0.5, 0.5))
    image.save(save_path)
    print(f"[SDXL Lightning] Image saved to {save_path}")
    return image

def download_audio(youtube_url: str) -> str:
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "quiet": True,
        "outtmpl": out,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"},
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    wav_path = str(Path(tmp) / next(p for p in os.listdir(tmp) if p.endswith(".wav") and "audio" in p))
    return wav_path, tmp  # Return both

def transcribe_wav(wav_path: str) -> str:
    def run_model(model_name, device):
        cmd = ["python", "yt2epub/utils/whisper_worker.py", model_name, wav_path, device]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        return result.stdout.strip()

    try:
        print("[Whisper] Trying base on GPU...")
        return run_model("base", "cuda")
    except Exception as e1:
        print(f"[Whisper base GPU Failed] {e1}")
        try:
            print("[Whisper] Trying tiny on GPU...")
            return run_model("tiny", "cuda")
        except Exception as e2:
            print(f"[Whisper tiny GPU Failed] {e2}")
            try:
                print("[Whisper] Trying tiny on CPU...")
                return run_model("tiny", "cpu")
            except Exception as e3:
                print(f"[Whisper CPU tiny Failed] {e3}")
                raise RuntimeError("❌ All Whisper fallbacks failed")

def generate_sd_prompt(book_text: str) -> tuple[str, str, str]:
    truncated = book_text[:6000]

    prompt = f"""You are an expert visual imagination assistant for book cover design.

First, generate a short, vivid, and symbolic **image prompt** for an AI image generator based on the following book content. 
Then, choose two suitable CSS-style hex colors for overlaying the book title:
- One for the **text fill**
- One for the **stroke outline**
Fill color should be a light color and stroke color should be a dark color. And they should look natural on the image if applied to the book title.

Rules:
- The image prompt should be anime-style, pastel, or illustrated.
- The image prompt should be less than 70 words long.
- Decide a specific color palette both with the image and the text colors.
- DO NOT include female characters or any text elements in the image.
- Output only the following block, no commentary:

---

IMAGE PROMPT:
<your image prompt here>

TEXT FILL COLOR:
<hex fill color like #ffffff>

STROKE COLOR:
<hex stroke color like #000000>

---

BOOK CONTENT:
{truncated}
"""

    try:
        chat = gemini_model.start_chat()
        response = chat.send_message(prompt)
        raw = response.text.strip()

        # Parsing
        lines = raw.splitlines()
        img_prompt, fill, stroke = "", "", ""
        current = None
        buffer = []

        for line in lines:
            line = line.strip()
            if line == "IMAGE PROMPT:":
                if current and buffer:
                    if current == "image": img_prompt = "\n".join(buffer).strip()
                current = "image"
                buffer = []
            elif line == "TEXT FILL COLOR:":
                if current and buffer:
                    if current == "image": img_prompt = "\n".join(buffer).strip()
                current = "fill"
                buffer = []
            elif line == "STROKE COLOR:":
                if current and buffer:
                    if current == "fill": fill = buffer[0].strip()
                current = "stroke"
                buffer = []
            elif current:
                buffer.append(line)

        # Final flush in case STROKE COLOR is last
        if current == "stroke" and buffer:
            stroke = buffer[0].strip()

        print("Gemini Response: ", raw)
        print("🎨 Gemini Image Prompt:", img_prompt)
        print("🎨 Text Fill Color:", fill)
        print("🎨 Stroke Color:", stroke)
        return img_prompt, fill, stroke

    except Exception as e:
        print(f"[Gemini Error in SD Prompt Generation] {e}")
        return "", DEFAULT_TEXT_COLOR, DEFAULT_STROKE_COLOR

def make_markdown_book_gemini(link: str, raw_text: str) -> str:
    """Uses Gemini to convert full transcript into a Markdown-structured book in one go."""
    prompt = f"""Task
Convert the entire spoken content of the transcript I supplied into a book‑like text ready for EPUB conversion.

Output rules

Return one fenced Markdown code block only—no extra commentary.

Inside the code block:

Use headings consistently:

    # (Heading 1) exclusively for the book title (decide a concise and good book title based on context, less than 7 words is better).

    ## (Heading 2) for chapters or main topic titles.

    ### (Heading 3) for subtopics or extremely impactful sentences/quotes.

    Avoid skipping heading levels (don't jump directly from # to ###).

Use plain paragraphs separated by a blank line for easy readability on eReaders. Keep paragraphs short and focused.

Use bold (**bold**) and italic (*italic*) sparingly, only to emphasize particularly important words or phrases.

Insert a horizontal rule (---) to indicate visual breaks, scene transitions, or major shifts in topics clearly.

Follow proper punctuation rules throughout. Apply correct capitalization, commas, periods, and other marks to make reading smooth and grammatically correct.

Divide the text under each topic or subtopic into multiple well‑balanced, readable paragraphs. Do not cram all content under one heading into a single paragraph—even if it stays on the same theme. Use natural breaks in thought, emphasis changes, or examples to clearly separate ideas.

Insert headings only at clear topic breaks after the preceding sentence has ended with punctuation. Never place a heading inside an unfinished sentence, inside quotation marks, or mid‑word.

Remove speech‑filler words such as "uh," "um," "er," "you know," "like," or "ee," while preserving the speaker’s tone and conversational flow.

Preserve the speaker’s natural phrasing and wording, but correct obvious transcript errors, such as misheard words, missing articles, awkward line breaks, or grammatical slips.

Keep the text complete and unabridged—no summaries, omissions, or cuts.

If the transcript lacks clear sections, determine logical topics, subtopics, and categories yourself and structure the text accordingly with appropriate headings.


Deliver the entire transcript in one message (as long as necessary; don’t truncate).


PurposeI’m on a full social‑media detox and prefer reading over watching. I will convert your Markdown directly to EPUB.


Important
    • Follow every instruction exactly—this prompt is self‑contained.
    • Output nothing outside the single Markdown code block.


--- BEGIN TRANSCRIPT ---
{raw_text}
--- END TRANSCRIPT ---
"""

    try:
        response = gemini_model.generate_content(prompt)
        return clean_markdown_fence(response.text)
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return ""

def clean_markdown_fence(text: str) -> str:
    """Strips ```markdown / ``` fences if present."""
    lines = text.strip().splitlines()
    if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text.strip()

def extract_title(raw: str) -> str:
    for ln in raw.splitlines():
        m = HEADER_RE.match(ln.strip())
        if m:
            return m.group(2).strip()
        if ln.strip():
            return re.split(r"[.?!]", ln.strip())[0][:80]
    return DEFAULT_TITLE

def slugify(text: str, maxlen: int = 60) -> str:
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^\w]+", "_", norm).strip("_")[:maxlen] or "book"

def prompt_from_text(title: str, raw: str) -> str: # Fallback image gen prompt generator function
    """
    Extracts headers from the raw text and incorporates them into the AI image generation prompt.
    """
    headers = []
    for line in raw.splitlines():
        match = HEADER_RE.match(line.strip())
        if match:
            headers.append(match.group(2).strip())

    # Join headers into a single string for the prompt
    headers_text = ", ".join(headers[:5])  # Limit to the first 5 headers for brevity

    return (
        f"an epic anime scene from a '{title}' anime, light pastel colors, anime scene, impressive image, vibrant, detailed, cinematic, themes: {headers_text}, "
        "NSFW, woman, girl, female, nudity, provocative, sexy, erotic, explicit, inappropriate"
    )

def generate_cover(prompt: str, path: str, title: str, text_color: str, stroke_color: str):
    """
    Generates a cover image. If the "Skip AI Image Generation" checkbox is selected,
    it generates a light gray gradient image instead of using AI-generated images.
    Overlays the book title in the center and saves to *path* (PNG).
    """

    # 1) Generate image --------------------------------------------------
    if os.path.exists(path):
        os.remove(path)

    if skip_ai_image_var.get():
        print("[AI Skip] Generating light gray gradient image")
        generate_light_gray_gradient_image(path)
    else:
        img = generate_cover_image(prompt, path)

    # 2) Overlay title text ---------------------------------------------------
    img = Image.open(path)
    draw = ImageDraw.Draw(img)

    # Try a nicer font if available, else fall back to default PIL bitmap font
    try:
        font = ImageFont.truetype("assets/Font.ttf", size=int(COVER_H * 0.1))
    except IOError:
        font = ImageFont.load_default()

    # Word‑wrap to ~80 % of image width
    max_w = int(COVER_W * 0.8)
    lines, words = [], title.split()
    while words:
        line = []
        while words and draw.textlength(" ".join(line + words[:1]), font) <= max_w:
            line.append(words.pop(0))
        if not line:  # a single oversized word
            line.append(words.pop(0))
        lines.append(" ".join(line))

    line_h = draw.textbbox((0, 0), "Ag", font)[3]
    total_h = line_h * len(lines)          #  keeps for later if you still need it
    y        = int(COVER_H * 0.08)         #  8 % down from the top edge

    # draw each wrapped line with white outline + black fill
    stroke = max(1, font.size // 25 + 1)          # ≈25 % of font size
    for ln in lines:
        w = draw.textlength(ln, font)
        x = (COVER_W - w) // 2
        draw.text((x, y), ln,
          font=font,
          fill=text_color,
          stroke_width=stroke,
          stroke_fill=stroke_color)
        line_gap = 3      #  ≈ 33 % of font height
        # or: line_gap = 6          #  fixed 6‑pixel gap
        y += line_h + line_gap

    # 3) Save & return ---------------------------------------------------------
    img.save(path)

def generate_light_gray_gradient_image(save_path: str):
    """Generates a light gray gradient image using Perlin noise."""
    width, height = COVER_W, COVER_H
    gradient = np.linspace(200, 240, width, dtype=np.uint8)  # Light gray gradient
    image = np.tile(gradient, (height, 1))
    img = Image.fromarray(image, mode="L").convert("RGB")
    img.save(save_path)
    print(f"[Gradient] Light gray gradient image saved to {save_path}")

# ─── EPUB builder ────────────────────────────────────────────────── #

def markdown_to_epub(
    raw_text,
    *,
    title,
    author,
    publisher,
    language,
    identifier,
    out_path,
    cover_path=None,
):
    book = epub.EpubBook()
    book.set_identifier(identifier or str(uuid.uuid4()))
    book.set_title(title)
    book.set_language(language)
    book.add_author(author)
    book.add_metadata("DC", "publisher", publisher)

    if cover_path and os.path.exists(cover_path):
        with open(cover_path, "rb") as c:
            book.set_cover("cover.png", c.read())

    chapters, cur_title, cur_body, cur_lvl = [], None, "", 1

    def flush():
        nonlocal cur_title, cur_body, cur_lvl
        if not cur_title:
            return
        tag = f"h{cur_lvl}"
        chap = epub.EpubHtml(
            title=cur_title,
            file_name=f"{slugify(cur_title)}.xhtml",
            content=f"<{tag}>{cur_title}</{tag}><p>{cur_body.strip().replace(chr(10), '</p><p>')}</p>",
        )
        book.add_item(chap)
        chapters.append(chap)
        cur_title, cur_body = None, ""

    for ln in raw_text.splitlines() + ["# END"]:
        m = HEADER_RE.match(ln.strip())
        if m:
            flush()
            cur_lvl = len(m.group(1))
            cur_title = m.group(2).strip()
        else:
            cur_body += ln + "\n"
    flush()

    book.toc, book.spine = chapters, ["nav"] + chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(out_path, book)

# ─── GUI ─────────────────────────────────────────────────────────── #

def launch():
    app = TkinterDnD.Tk()
    app.title("📖 Text → EPUB (AI cover)")
    app.geometry("900x600")  # wider window to accommodate side preview
    app.configure(bg="#f5f5f5")
    
    # Additional batch input field
    batch_frame = tk.Frame(app, bg="#f5f5f5")
    batch_frame.pack(fill="x", padx=8, pady=(4, 4))

    tk.Label(batch_frame, text="Link(s*) (*each should be in new lines):", bg="#f5f5f5").pack(anchor="w")
    batch_entry = tk.Text(batch_frame, height=3, wrap="word")
    batch_entry.pack(fill="x", expand=True)

    tk.Button(batch_frame, text="🚀 Process All", bg="#009688", fg="white",
            command=lambda: start_batch_process(batch_entry.get("1.0", "end"))).pack(anchor="e", pady=(2, 0))

    # Add the checkbox to the GUI
    fast_transcription_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        batch_frame,
        text="Fast Transcription",
        variable=fast_transcription_var,
        bg="#f5f5f5"
    ).pack(anchor="w")

    # Add the checkbox to the GUI
    global skip_ai_image_var
    skip_ai_image_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        batch_frame,
        text="Skip AI Image Generation",
        variable=skip_ai_image_var,
        bg="#f5f5f5"
    ).pack(anchor="w")

    # Add a checkbox for "Open Play Books after generation"
    open_play_books_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        batch_frame,
        text="📖 Open Play Books after generation",
        variable=open_play_books_var,
        bg="#f5f5f5"
    ).pack(anchor="w")

    # Add a checkbox for "Upload to Google Drive"
    upload_to_drive_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        batch_frame,
        text="☁️ Upload to Google Drive",
        variable=upload_to_drive_var,
        bg="#f5f5f5"
    ).pack(anchor="w")

    # ── Metadata frame ── #
    meta_container = tk.Frame(app, bg="#f5f5f5")
    meta_container.pack(fill="x", padx=8, pady=(6, 0))

    # Toggle button
    def toggle_meta():
        if meta.winfo_viewable():
            meta.pack_forget()
            toggle_btn.config(text="🔽 Show Metadata")
        else:
            meta.pack(fill="x", padx=8, pady=(4, 4))
            toggle_btn.config(text="🔼 Hide Metadata")

    toggle_btn = tk.Button(meta_container, text="🔽 Show Metadata", command=toggle_meta, bg="#dddddd")
    toggle_btn.pack(anchor="w", pady=(0, 4))

    # Metadata section (initially hidden)
    meta = tk.LabelFrame(meta_container, text="Metadata", bg="#f5f5f5", padx=6, pady=4)
    # meta.pack(...)  ← don't pack it yet — it starts collapsed

    def add_row(lbl, default="", width=44):
        tk.Label(meta, text=lbl, bg="#f5f5f5").grid(sticky="w")
        e = tk.Entry(meta, width=width)
        e.insert(0, default)
        e.grid(row=add_row.row, column=1, padx=4, pady=2, sticky="we")
        add_row.row += 1
        return e

    add_row.row = 0
    title_e = add_row("Title:", DEFAULT_TITLE)
    author_e = add_row("Author:", DEFAULT_AUTHOR)
    publ_e = add_row("Publisher:", DEFAULT_PUBLISHER)
    lang_e = add_row("Language:", DEFAULT_LANGUAGE)
    id_e = add_row("Identifier:", "")
    file_e = add_row("Output name:", DEFAULT_OUT_NAME)
    meta.grid_columnconfigure(1, weight=1)

    # ── Central content frame (text left, preview right) ── #
    content = tk.Frame(app, bg="#f5f5f5")
    content.pack(fill="both", expand=True, padx=8, pady=4)
    content.grid_columnconfigure(0, weight=1)
    content.grid_columnconfigure(1, weight=0)
    content.grid_rowconfigure(0, weight=1)

    # Add a toggle button to collapse/expand the Markdown textbox

    def toggle_markdown():
        if text_frame.winfo_viewable():
            text_frame.pack_forget()
            toggle_markdown_btn.config(text="🔽 Show Markdown")
        else:
            text_frame.pack(fill="both", expand=True)
            toggle_markdown_btn.config(text="🔼 Hide Markdown")

    # Add the toggle button above the Markdown textbox
    toggle_markdown_btn = tk.Button(content, text="🔽 Show Markdown", command=toggle_markdown, bg="#dddddd")
    toggle_markdown_btn.grid(row=0, column=0, sticky="w", pady=(0, 4))

    # Text area on the left (shrinkable)
    text_frame = tk.Frame(content, bg="#f5f5f5")
    text_frame.grid(row=0, column=0, sticky="nsew")
    tk.Label(text_frame, text="Paste Markdown here ↓", bg="#f5f5f5").pack(anchor="w")
    txt_box = tk.Text(text_frame, wrap="word")
    txt_box.pack(fill="both", expand=True)

    # Initially hide the Markdown textbox
    text_frame.pack_forget()

    # Right panel (fixed size)
    right_panel = tk.Frame(content, bg="#f5f5f5", width=300)
    right_panel.grid(row=0, column=1, sticky="ns")
    right_panel.grid_propagate(False)  # Prevent auto-resize

    status = tk.Label(right_panel, text="No cover yet", bg="#f5f5f5", fg="gray")
    status.pack(pady=(0, 4))
    
    progress = tk.Label(right_panel, text="", bg="#f5f5f5", fg="gray")
    progress.pack(pady=(0, 4))

    # --- Custom prompt box -------------------------------------------
    tk.Label(right_panel, text="Custom image generation prompt (optional)", bg="#f5f5f5").pack()
    prompt_entry = tk.Text(right_panel, height=3, width=30, wrap="word")
    prompt_entry.pack(pady=(0, 4))


    cover_img_tk = None
    cover_label = tk.Label(right_panel, bg="#f5f5f5")
    cover_label.pack()

    # Buttons under preview
    btn_frame = tk.Frame(right_panel, bg="#f5f5f5")
    
    # Text and stroke color pickers
    text_color = DEFAULT_TEXT_COLOR
    stroke_color = DEFAULT_STROKE_COLOR

    btn_frame.pack(pady=6)

    # ── Helper functions ── #
    def collect_meta():
        t = title_e.get().strip() or DEFAULT_TITLE
        outnm = file_e.get().strip() or DEFAULT_OUT_NAME
        if not outnm.lower().endswith(".epub"):
            outnm += ".epub"
        full_path = os.path.join(OUTPUT_DIR, outnm)
        return dict(
            title=t,
            author=author_e.get().strip() or DEFAULT_AUTHOR,
            publisher=publ_e.get().strip() or DEFAULT_PUBLISHER,
            language=lang_e.get().strip() or DEFAULT_LANGUAGE,
            identifier=id_e.get().strip() or None,
            out_path=full_path,
        )


    def load_youtube(url):
        print("[0] Start YouTube load")
        status.config(text="⬇️  Downloading...", fg="blue")

        try:
            if fast_transcription_var.get():
                print("[1] Using Fast Transcription API")
                video_id = re.search(r"(?:v=|be/|embed/|youtu.be/)([\w-]{11})", url).group(1)
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = "\n".join([entry['text'] for entry in transcript])
                print(f"[1✓] Transcript length: {len(transcript_text)} chars")
            else:
                print("[1] Calling download_audio()")
                status.config(text="⬇️  Downloading...", fg="blue")
                wav, tmpdir = download_audio(url)
                print(f"[1✓] Audio downloaded → {wav}")

                print("[2] Calling transcribe_wav()")
                status.config(text="🗣️  Transcribing...", fg="blue")
                transcript_text = transcribe_wav(wav)
                shutil.rmtree(tmpdir, ignore_errors=True)  # cleanup after use
                print(f"[2✓] Transcript length: {len(transcript_text)} chars")
        except Exception as e:
            print(f"[1✗] Failed during transcription: {e}")
            status.config(text="❌ Transcription failed", fg="red")
            return

        try:
            print("[3] Calling make_full_markdown_book()")
            status.config(text="📖 Generating Markdown...", fg="blue")
            md = make_markdown_book_gemini(url, transcript_text)
            print(f"[3✓] Markdown length: {len(md)} chars")
        except Exception as e:
            print(f"[3✗] Failed during Markdown generation: {e}")
            status.config(text="❌ Markdown failed", fg="red")
            return

        try:
            print("[4] Writing into txt_box")
            txt_box.delete("1.0", "end")
            txt_box.insert("1.0", md)
            print("[4✓] Text inserted into GUI")
        except Exception as e:
            print(f"[4✗] Failed to insert text: {e}")
            status.config(text="❌ GUI insert failed", fg="red")
            return

        try:
            print("[5] Generating cover")
            status.config(text="🎨 Generating cover...", fg="blue")
            generate_clicked()
            print("[5✓] Cover generated successfully")
        except Exception as e:
            print(f"[5✗] Failed during cover generation: {e}")
            status.config(text="❌ Cover generation failed", fg="red")
            return
        
        try:
            print("[6] Converting and saving EPUB")
            status.config(text="📖 Saving EPUB...", fg="blue")
            convert_clicked()
            print("[6✓] EPUB saved successfully")
        except Exception as e:
            print(f"[6✗] Failed during EPUB conversion: {e}")
            status.config(text="❌ EPUB conversion failed", fg="red")

        status.config(text="✅ The process is completed!", fg="green")
        print("[7✓] All steps completed")

    def start_batch_process(raw_links: str):
        links = [url.strip() for url in raw_links.splitlines() if url.strip()]
        if not links:
            messagebox.showwarning("Empty", "Please enter at least one YouTube link.")
            return

        threading.Thread(target=lambda: run_batch_links(links), daemon=True).start()

    def run_batch_links(links: list[str]):
        for i, url in enumerate(links, 1):
            start_time = time.time()
            
            print(f"\n===== Processing {i}/{len(links)}: {url} =====")
            status.config(text=f"🎬 Processing {i}/{len(links)}", fg="blue")
            
            load_youtube(url)
            status.config(text=f"✅ Processed {i}/{len(links)}", fg="green")
            
            # Wait for remaining time to hit 60 seconds
            # elapsed = time.time() - start_time
            # if elapsed < 60:
            #     remaining = 60 - elapsed
            #     print(f"🕐 Waiting {int(remaining)}s to respect cooldown...")
            #     time.sleep(remaining)


    def autofill(raw):
        title_e.delete(0, "end")
        title_e.insert(0, extract_title(raw))
        
        file_e.delete(0, "end")
        file_e.insert(0, slugify(title_e.get()) + ".epub")

    def show_cover(path):
        nonlocal cover_img_tk
        img = Image.open(path)
        img.thumbnail((256, 384))  # preview fits label size
        cover_img_tk = ImageTk.PhotoImage(img)
        cover_label.config(image=cover_img_tk)

    def generate_clicked():
        nonlocal text_color, stroke_color

        raw = clean_markdown_fence(txt_box.get("1.0", "end"))
        if not raw:
            return messagebox.showwarning("Empty", "Paste or load text first!")
        autofill(raw)

        if not skip_ai_image_var.get():
            if prompt_entry.get("1.0", "end").strip():
                prompt = prompt_entry.get("1.0", "end").strip()
            else:
                try:
                    prompt, ai_fill, ai_stroke = generate_sd_prompt(raw)

                    # Override colors if provided by AI
                    text_color = ai_fill or DEFAULT_TEXT_COLOR
                    stroke_color = ai_stroke or DEFAULT_STROKE_COLOR
                    if not prompt or len(prompt) < 5:
                        raise ValueError("Empty or weak AI prompt")
                except Exception as e:
                    print(f"[AI Prompt Error] Falling back. Reason: {e}")
                    prompt = prompt_from_text(title_e.get(), raw)
        else:
            prompt = prompt_entry.get("1.0", "end").strip() or "a light gray gradient background"

        try:
            generate_cover(prompt, COVER_FN, title_e.get(), text_color, stroke_color)
            status.config(text="✅ Cover generated", fg="green")
            show_cover(COVER_FN)
        except Exception as e:
            status.config(text=f"❌ Cover failed: {e}", fg="red")
            if os.path.exists(COVER_FN):
                os.remove(COVER_FN)
            cover_label.config(image="")

    def convert_clicked():
        raw = clean_markdown_fence(txt_box.get("1.0", "end"))
        if not raw:
            return messagebox.showwarning("Empty", "Paste or load text first!")
        autofill(raw)
        meta = collect_meta()
        try:
            markdown_to_epub(raw, cover_path=COVER_FN if os.path.exists(COVER_FN) else None, **meta)
            print("Done", f"📚 EPUB saved:\n{meta['out_path']}")

            # Open Play Books and highlight the EPUB file if the checkbox is checked
            if open_play_books_var.get():
                try:
                    # Open Google Play Books in the default browser
                    webbrowser.open("https://play.google.com/books")
                    print("[Play Books] Opened Google Play Books in browser.")

                    # Highlight the generated EPUB file in the file explorer
                    epub_path = meta['out_path']
                    folder_path = os.path.dirname(os.path.abspath(epub_path))
                    os.startfile(folder_path, 'explore')
                    print(f"[File Explorer] Highlighted EPUB file: {epub_path}")
                except Exception as e:
                    print(f"[Error] Failed to open Play Books or highlight file: {e}")

            # Upload to Google Drive if the checkbox is checked
            if upload_to_drive_var.get():
                try:
                    epub_path = meta['out_path']
                    result = subprocess.run(["gdrive", "upload", epub_path], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"[Google Drive] Upload successful: {result.stdout.strip()}")
                    else:
                        print(f"[Google Drive] Upload failed: {result.stderr.strip()}")
                except Exception as e:
                    print(f"[Error] Failed to upload to Google Drive: {e}")
        except Exception as e:
            print("Error", str(e))

    tk.Button(
        btn_frame,
        text="🎨 Generate Cover",
        bg="#2196f3",
        fg="white",
        width=18,
        command=generate_clicked,
    ).grid(row=0, column=0, padx=4, pady=2)

    tk.Button(
        btn_frame,
        text="📖 Save as EPUB",
        bg="#4caf50",
        fg="white",
        width=18,
        command=convert_clicked,
    ).grid(row=1, column=0, padx=4, pady=2)

    app.mainloop()

if __name__ == "__main__":
    launch()