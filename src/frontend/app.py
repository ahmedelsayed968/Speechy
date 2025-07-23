import streamlit as st
import requests
import os
from dotenv import load_dotenv
load_dotenv()
FASTAPI_URL = os.environ.get("SERVER_URL")   # Change if deployed elsewhere

st.title("🗣️ Speechy: Gender & Speech Detection")

st.markdown("Upload a **WAV file** (16kHz, mono, 16-bit PCM) to detect if it contains speech and predict the speaker's gender.")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    with st.spinner("Uploading and analyzing file..."):

        # Step 1: Upload file to FastAPI
        files = {"file": (uploaded_file.name, uploaded_file, "audio/wav")}
        upload_response = requests.post(f"{FASTAPI_URL}/Speechy/upload_file", files=files)

        if upload_response.status_code == 200:
            data = upload_response.json()
            st.success("✅ File uploaded successfully!")

            st.markdown(f"**Sample Rate:** {data['sample_rate']}")
            st.markdown(f"**File ID:** `{data['file_id']}`")

            # Step 2: Call Speech analysis
            st.info("Running VAD + Gender classification...")
            response = requests.get(f"{FASTAPI_URL}/Speech/", params={"file_id": data["file_id"]})

            if response.status_code == 200:
                result = response.json()
                if result["speech"]:
                    st.success(f"🟢 Speech Detected\n\n**Gender**: `{result['gender']}`")
                else:
                    st.warning("🟡 No speech detected.")
            else:
                st.error(f"❌ Error analyzing file: {response.text}")
        else:
            st.error(f"❌ Upload failed: {upload_response.text}")
