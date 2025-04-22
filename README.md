![Python](https://img.shields.io/badge/Python-3.13+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> “Discipline equals freedom.”  
> — Jocko Willink

This tool is a great way to reduce passive screen time and shift your YouTube consumption into a more mindful, readable format.

# 📖 yt2epub — YouTube ➜ EPUB AI Converter

Transform YouTube videos into beautiful, ready-to-read eBooks (EPUB format) with AI-generated covers — in just one click.  
No overstimulation anymore. Just clean, value-focused, structured reading with a shelf-ready aesthetic.

---

### 🚀 Features

- 🎙️ Converts **YouTube video speech** into text using **Whisper AI** or **YouTube Transcript API** (Fast Transcription mode)
- ✍️ Automatically turns speech into a **structured Markdown book** with headings, paragraphs, and natural flow using **Gemini**
- 🎨 Uses the book's theme to generate an **AI cover image** (via SDXL) with **customized text colors**, or generates a **light gray gradient cover** (Skip AI Image Generation mode)
- 📘 Packages everything into an **EPUB** file — perfect for e-readers like Google Play Books, Kindle (via Calibre), or tablets
- ☁️ **Optional Google Drive upload**: Automatically uploads the generated EPUB to your Google Drive
- 🔁 Includes batch processing: drop multiple links, get multiple books
- 📖 **Play Books integration**: Automatically opens the EPUB in Google Play Books and highlights it in the file explorer
- 🌐 **Translation Support**: Translate the transcript into any language before generating the book (default: Türkçe)
- ✅ **Customizable Text Alignment**: Overlay text on the cover is now aligned to the left for a cleaner look

---

### 🧠 How it works (Under the hood)

1. Paste your YouTube video links (one per line)
2. 🧲 Downloads audio via `yt-dlp` or fetches the transcript via **YouTube Transcript API** (Fast Transcription mode)
3. 🧠 Transcribes it with **Whisper AI** (if not using Fast Transcription)
4. 🧠 Sends the transcript to **Gemini (Google's AI)** to generate structured Markdown
5. 🌐 Optionally translates the transcript into your chosen language using Gemini
6. 🎨 Sends the text back to Gemini to create a cover image prompt + suitable colors
7. 🖼️ Uses **Stable Diffusion XL Lightning** locally to generate the cover image, or generates a light gray gradient cover (Skip AI Image Generation mode)
8. ✍️ Adds the title to the image with color contrast and left-aligned text
9. 📦 Converts everything to an EPUB file and saves it in the `Outputs/` folder
10. ☁️ Optionally uploads the EPUB to Google Drive
11. 📖 Optionally opens the EPUB in Google Play Books and highlights it in the file explorer
12. ✅ Moves on to the next link, if any

---

### 💻 Installation

```bash
git clone https://github.com/yourname/yt2epub
cd yt2epub
pip install -e .
```

You’ll also need to download:

- **Stable Diffusion XL Lightning** checkpoint:  
  `ByteDance/SDXL-Lightning → sdxl_lightning_4step_unet.safetensors`

- **Base model**:  
  `stabilityai/stable-diffusion-xl-base-1.0`

Put both inside a local folder (e.g., `models/`)

---

### 🔑 Google Gemini API Key

This app uses **Gemini** for book structure, translation, and cover prompt generation.  
You **must provide your own key**.

#### Add it in a `.env` file:

```
GOOGLE_API_KEY=your-api-key-here
```

Or hardcode it (not recommended for public use):

```python
genai.configure(api_key="YOUR_API_KEY_HERE")
```

---

### 🧪 Dependencies

Included in `setup.py`, but for reference:

- `yt_dlp`
- `faster_whisper`
- `google-generativeai`
- `diffusers`, `torch`, `transformers`
- `ebooklib`
- `Pillow`
- `tkinter`, `tkinterdnd2`
- `llama-cpp-python` (optional)
- `safetensors`, `huggingface_hub`
- `python-dotenv`

---

### 🧰 Usage (CLI)

Once installed, run:

```bash
yt2epub
```

Then:
- Paste one or more YouTube links into the input box
- Optionally edit the title/author
- Use the **Translate to** checkbox to translate the transcript into your desired language
- Hit **"Process All"**
- Let the AI cook 🧠🔥

After processing, your EPUBs will appear in the `Outputs/` folder — complete with clean structure and cover art.

---

### ✨ Output Example

```plaintext
Outputs/
└── your_book_title.epub
```

Looks awesome on:
- Google Play Books ✅
- Kobo & Kindle (via Calibre) ✅
- Any EPUB-supported reader ✅

---

### 📌 Notes

- Requires a decent GPU for Stable Diffusion (16 GB+ VRAM is ideal)
- Whisper runs on CPU by default but tries GPU if available
- Gemini may throw occasional API limits — consider batching responsibly
- **Fast Transcription mode**: Skips audio download and Whisper transcription, directly fetching the transcript via YouTube Transcript API (if available)
- **Skip AI Image Generation mode**: Skips AI-based cover generation and uses a light gray gradient cover instead
- **Translation Support**: Translate the transcript into any language before generating the book (default: Türkçe)
- **Customizable Text Alignment**: Overlay text on the cover is now aligned to the left for a cleaner look
- **Google Drive upload**: Requires `gdrive` CLI tool to be installed and authenticated
- **Play Books integration**: Automatically opens the EPUB in Google Play Books and highlights it in the file explorer for easy drag-and-drop

---

### 💡 Tips

- Want to customize the image prompt? Type your own in the box before hitting Generate Cover
- You can skip cover generation if you only want Markdown ➜ EPUB
- Book metadata (title, author, etc.) can be edited before conversion
- Use the "Fast Transcription" checkbox for quicker processing when YouTube transcripts are available
- Use the "Skip AI Image Generation" checkbox to save GPU resources and generate a simple gradient cover
- Use the "Translate to" checkbox to translate the transcript into your desired language before processing
- Enable or disable "Upload to Google Drive" and "Open Play Books after generation" based on your workflow

---

### 🧼 Optional Cleanup

Remove unused imports to slim down the code (see `main.py` top section).

---

### 🧑‍💻 Author

Created by **Mahmud Esad Yazar**  

---