import whisper, pathlib

model = whisper.load_model("large-v3")
result = model.transcribe(
    "/scratch/data/m22aie218/Speech-Understanding/assignment2/data/original_segment.wav",
    language="en"
)
ref_path = "/scratch/data/m22aie218/Speech-Understanding/assignment2/data/reference_transcript.txt"
pathlib.Path(ref_path).write_text(result["text"].strip())
print("Reference saved:", result["text"][:200])