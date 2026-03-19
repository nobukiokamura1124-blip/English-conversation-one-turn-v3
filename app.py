import streamlit as st
import wave
import tempfile
import numpy as np
from gtts import gTTS
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue

# -----------------------------
# OpenAI
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("AI英会話（1ターン）")

# -----------------------------
# 音声保存用
# -----------------------------
if "audio_frames" not in st.session_state:
    st.session_state.audio_frames = []

# -----------------------------
# WebRTC（★STUN追加済み）
# -----------------------------
ctx = webrtc_streamer(
    key="one-turn",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)

# -----------------------------
# 音声取得
# -----------------------------
if ctx.audio_receiver:
    try:
        frames = ctx.audio_receiver.get_frames(timeout=0.2)
        for frame in frames:
            sound = frame.to_ndarray()

            if sound.ndim == 2:
                sound = sound.mean(axis=1)

            if sound.dtype != np.int16:
                sound = sound.astype(np.int16)

            st.session_state.audio_frames.append(sound)

    except queue.Empty:
        pass

# -----------------------------
# ボタン
# -----------------------------
if st.button("録音終了 → AIに送る"):
    if len(st.session_state.audio_frames) == 0:
        st.warning("音声が取れてない。マイクON確認")
    else:
        try:
            # WAV化
            audio_data = np.concatenate(st.session_state.audio_frames, axis=0)

            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

            with wave.open(tmp_wav.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(audio_data.tobytes())

            # 音声 → テキスト
            with open(tmp_wav.name, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=f
                )

            user_text = transcript.text
            st.write("🧑‍💬", user_text)

            # AI応答
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly English conversation partner. Keep replies short and natural."
                    },
                    {
                        "role": "user",
                        "content": user_text
                    }
                ]
            )

            reply = response.choices[0].message.content
            st.write("🤖", reply)

            # 音声出力
            tmp_mp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")

            tts = gTTS(text=reply, lang="en")
            tts.save(tmp_mp3.name)

            st.audio(tmp_mp3.name)

            # リセット
            st.session_state.audio_frames = []

        except Exception as e:
            st.error(f"エラー: {e}")