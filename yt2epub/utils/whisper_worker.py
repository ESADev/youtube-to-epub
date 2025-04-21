# whisper_worker.py
import sys
from faster_whisper import WhisperModel

model_name = sys.argv[1]  # e.g., "base.en"
audio_path = sys.argv[2]
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"  # "cuda" or "cpu"

model = WhisperModel(model_name, device=device)

segments, _ = model.transcribe(audio_path, beam_size=5, vad_filter=True)
text = " ".join([s.text.strip() for s in segments])
sys.stdout.reconfigure(encoding='utf-8')
print(text)