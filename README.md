![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> “Discipline equals freedom.”  
> — Jocko Willink

This tool is a great way to reduce passive screen time and shift your YouTube consumption into a more mindful, readable format.

# 📖 YouTube ➜ EPUB AI Converter

Transform YouTube videos into beautiful, ready-to-read eBooks (EPUB format) with AI-generated covers — in just one click.  
No gray rectangles in your Play Books anymore. Just clean, structured reading with a shelf-ready aesthetic.

---

### 🚀 Features

- 🎙️ Converts **YouTube video speech** into text using **Whisper AI**
- ✍️ Automatically turns speech into a **structured Markdown book** with headings, paragraphs, and natural flow using **Gemini**
- 🎨 Uses the book's theme to generate an **AI cover image** (via SDXL) with **customized text colors**
- 📘 Packages everything into an **EPUB** file — perfect for e-readers like Google Play Books, Kindle (via Calibre), or tablets

---

### 🧠 How it works (Under the hood)

1. Paste your YouTube video links (one per line)
2. 🧲 Downloads audio via `yt-dlp`
3. 🧠 Transcribes it with **Whisper AI**
4. 🧠 Sends the transcript to **Gemini (Google's AI)** to generate structured Markdown
5. 🎨 Sends the text back to Gemini to create a cover image prompt + suitable colors
6. 🖼️ Uses **Stable Diffusion XL Lightning** locally to generate the cover image
7. ✍️ Adds the title to the image with color contrast
8. 📦 Converts everything to an EPUB file and saves it in the `Outputs/` folder
9. ✅ Moves on to the next link, if any

---

### 💻 Installation

```bash
git clone https://github.com/yourname/youtube-to-epub-ai
cd youtube-to-epub-ai
pip install -r requirements.txt
```

You’ll also need to download:

- **Stable Diffusion XL Lightning** checkpoint:  
  `ByteDance/SDXL-Lightning → sdxl_lightning_4step_unet.safetensors`

- **Base model**:  
  `stabilityai/stable-diffusion-xl-base-1.0`

Put both inside a local folder (e.g., `D:/AI/models/`)

---

### 🔑 Google Gemini API Key

This app uses **Gemini** for book structure and cover prompt generation.  
You **must provide your own key**.

#### Create a `.env` file or set it in code:

```bash
export GOOGLE_API_KEY=your-key-here
```

Or hardcode it (not recommended for public use):

```python
genai.configure(api_key="YOUR_API_KEY_HERE")
```

---

### 🧪 Dependencies

- `yt_dlp`
- `faster_whisper`
- `google-generativeai`
- `diffusers`, `torch`, `transformers`
- `ebooklib`
- `Pillow`
- `tkinter`, `tkinterdnd2`
- `llama-cpp-python` (optional)
- `safetensors`, `huggingface_hub`

---

### 🧰 Usage

```bash
python main.py
```

Then:
- Paste one or more YouTube links into the input box
- Optionally edit the title/author
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

---

### 💡 Tips

- Want to customize the image prompt? Type your own in the box before hitting Generate Cover
- You can skip cover generation if you only want Markdown ➜ EPUB
- Book metadata (title, author, etc.) can be edited before conversion

---

### 🧼 Optional Cleanup

Remove unused imports to slim down the code (see `main.py` top section).

---

### 🧑‍💻 Author

Created by **Mahmud Esad Yazar**  

---