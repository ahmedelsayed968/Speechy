from pathlib import Path
import shutil
from fastapi import FastAPI, UploadFile, File,HTTPException
from typing import Annotated
import numpy as np
from scipy.io.wavfile import write , read as read_wav
from uuid import uuid4
from config.paths import PROJECT_ROOT
from gender_detector.ecapa import ECAPADetectorStrategy
import torchaudio
from api.schemas import SpeechAnalysisResponse,UploadResponse
app =  FastAPI()
from gender_detector.speechy import SpeechyVoiceGenderDetectionService, SpeecyModelResponse

voice_detector_service = SpeechyVoiceGenderDetectionService()

UPLOADED_FILES_DIR= PROJECT_ROOT/"uploaded_files"
UPLOADED_FILES_DIR.mkdir(exist_ok=True)

@app.get("/")
async def home():
    return {
        "message":"hello from speechy!"
    }


@app.post("/Speechy/upload_file",response_model=UploadResponse)
async def create_file(file: Annotated[UploadFile, File()],):
    if file.content_type not in ["audio/wav", "audio/x-wav"]:
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    # Save uploaded file to a temp location
    temp_path = Path("temp_upload.wav")
    with temp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        audio_tensor = ECAPADetectorStrategy.load_audio(temp_path)

        file_id = f"{file.filename}_{str(uuid4())}"
        file_to_save = UPLOADED_FILES_DIR / f"{file_id}.wav"
        torchaudio.save(file_to_save,
                        audio_tensor,
                        sample_rate=16000,
                        encoding="PCM_S",          # Signed 16-bit PCM
                        bits_per_sample=16
                 )
        return UploadResponse(file_id=file_id,message="WAV file processed",sample_rate=16000)

    finally:
        temp_path.unlink(missing_ok=True)  # Clean up temp file

@app.get("/Speech/",response_model=SpeecyModelResponse)
async def process_file(file_id:str):
    try:
        full_path = UPLOADED_FILES_DIR / f"{file_id}.wav"
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="No file with this ID")
        return voice_detector_service.predict(audio_path=full_path)
    except Exception as e:
        print("Error occurred:", e)
        raise HTTPException(status_code=500, detail=str(e))

