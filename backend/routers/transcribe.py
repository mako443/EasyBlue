from typing import Annotated
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import HTMLResponse
import whisper
import torchaudio
import subprocess
import os.path as osp
import os
import numpy as np

LANGUAGES = {
    "english": "en",
    "chinese": "zh",
    "german": "de",
    "spanish": "es",
    "russian": "ru",
    "korean": "ko",
    "french": "fr",
    "japanese": "ja",
    "portuguese": "pt",
    "turkish": "tr",
    "polish": "pl",
    "catalan": "ca",
    "dutch": "nl",
    "arabic": "ar",
    "swedish": "sv",
    "italian": "it",
    "indonesian": "id",
    "hindi": "hi",
    "finnish": "fi",
    "vietnamese": "vi",
    "hebrew": "iw",
    "ukrainian": "uk",
    "greek": "el",
    "malay": "ms",
    "czech": "cs",
    "romanian": "ro",
    "danish": "da",
    "hungarian": "hu",
    "tamil": "ta",
    "norwegian": "no",
    "thai": "th",
    "urdu": "ur",
    "croatian": "hr",
    "bulgarian": "bg",
    "lithuanian": "lt",
    "latin": "la",
    "maori": "mi",
    "malayalam": "ml",
    "welsh": "cy",
    "slovak": "sk",
    "telugu": "te",
    "persian": "fa",
    "latvian": "lv",
    "bengali": "bn",
    "serbian": "sr",
    "azerbaijani": "az",
    "slovenian": "sl",
    "kannada": "kn",
    "estonian": "et",
    "macedonian": "mk",
    "breton": "br",
    "basque": "eu",
    "icelandic": "is",
    "armenian": "hy",
    "nepali": "ne",
    "mongolian": "mn",
    "bosnian": "bs",
    "kazakh": "kk",
    "albanian": "sq",
    "swahili": "sw",
    "galician": "gl",
    "marathi": "mr",
    "punjabi": "pa",
    "sinhala": "si",
    "khmer": "km",
    "shona": "sn",
    "yoruba": "yo",
    "somali": "so",
    "afrikaans": "af",
    "occitan": "oc",
    "georgian": "ka",
    "belarusian": "be",
    "tajik": "tg",
    "sindhi": "sd",
    "gujarati": "gu",
    "amharic": "am",
    "yiddish": "yi",
    "lao": "lo",
    "uzbek": "uz",
    "faroese": "fo",
    "haitian creole": "ht",
    "pashto": "ps",
    "turkmen": "tk",
    "nynorsk": "nn",
    "maltese": "mt",
    "sanskrit": "sa",
    "luxembourgish": "lb",
    "myanmar": "my",
    "tibetan": "bo",
    "tagalog": "tl",
    "malagasy": "mg",
    "assamese": "as",
    "tatar": "tt",
    "hawaiian": "haw",
    "lingala": "ln",
    "hausa": "ha",
    "bashkir": "ba",
    "javanese": "jw",
    "sundanese": "su",
}


def load_audio(path):
    if not osp.isfile(path):
        raise ValueError(f"File {path} does not exist")

    filename = osp.basename(path)
    filename = osp.splitext(filename)[0]
    path_converted = "../files/" + filename + ".mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-c:v", "copy", "-c:a", "libmp3lame", "-q:a", "4", path_converted],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not osp.isfile(path_converted):
        raise ValueError(f"Conversion failed.")

    waveform, sample_rate = torchaudio.load(path_converted)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

    os.remove(path_converted)

    return waveform.squeeze(0)


model = None
router = APIRouter()


@router.post("/transcribe")
def transcribe(language: str, files: Annotated[list[bytes], File()]):
    global model
    if model is None:
        model = whisper.load_model("base")

    language = language.lower()
    if language not in LANGUAGES:
        return {"error": "Language not supported"}
    language = LANGUAGES[language]

    file = files[0]
    filepath = f"../files/{np.random.randint(100000)}.m4a"
    with open(filepath, "wb") as f:
        f.write(file)

    audio = load_audio(filepath)
    if audio is None:
        return {"error": "Could not load audio"}

    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    configuration = {"beam_size": 5, "fp16": False, "task": "transcribe", "language": language}
    options = whisper.DecodingOptions(**configuration)

    result = whisper.decode(model, mel, options)
    text = result.text
    return {"text": text}


@router.post("/files/")
async def create_files(files: Annotated[list[bytes], File()]):
    file = files[0]
    filepath = f"../files/{np.random.randint(100000)}.m4a"
    with open(filepath, "wb") as f:
        f.write(file)
    return {"status": "ok"}


@router.post("/uploadfiles/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}


@router.get("/upload-test")
async def main():
    content = """
<body>
<form action="/transcribe?language=english" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
    """
    return HTMLResponse(content=content)
