from faster_whisper import WhisperModel

# Load Faster-Whisper model globally (smaller models = faster on CPU)
model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio file using Faster-Whisper (local).
    Returns raw transcript text.
    """
    segments, _ = model.transcribe(file_path)
    transcript = " ".join([segment.text for segment in segments])
    return transcript

def clean_transcript(text: str) -> str:
    """
    Optional: Clean transcript by removing fillers and repetitions.
    """
    fillers = ["um", "uh", "like", "you know"]
    for f in fillers:
        text = text.replace(f, "")
    return text.strip()
