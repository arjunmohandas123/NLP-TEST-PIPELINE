"""
Voice pipeline test harness:
  mic capture -> VAD -> DeepFilterNet denoise -> speaker verification gate
  -> Whisper ASR -> confidence gate

Two independent gates decide whether anything reaches your NLP/robot logic:
  1. Speaker verification -- is this YOUR voice? (rejects other speakers)
  2. Confidence gate       -- is the transcript reliable? (rejects uncertainty)
Both must pass before a command is treated as valid.

Run on a laptop with a headset or Bluetooth microphone connected. This validates the SOFTWARE
CHAIN, not acoustic robustness -- see the caveats about Bluetooth mic quality
discussed alongside this script.

Usage:
    python audio_pipeline.py --list-devices
    python audio_pipeline.py --device 3 --enroll          # one-time: record your voiceprint
    python audio_pipeline.py --device 3                   # normal run: verify + transcribe
    python audio_pipeline.py --device 3 --no-denoise       # A/B test vs raw audio
    python audio_pipeline.py --device 3 --no-speaker-check # disable speaker gate (debugging)
"""

import argparse
import queue
import sys
import time

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

VOICEPRINT_PATH = "voiceprint.npy"
SPEAKER_SIMILARITY_THRESHOLD = 0.55  # cosine similarity below this -> reject speaker

# ---- Config -----------------------------------------------------------
MIC_SAMPLE_RATE = 16000     # what we ask the OS for (Bluetooth headset mics usually max out here anyway)
DF_SAMPLE_RATE = 48000      # DeepFilterNet's native rate
WHISPER_SAMPLE_RATE = 16000
FRAME_MS = 30               # VAD frame size
SILENCE_TIMEOUT_S = 1.0     # stop recording a segment after this much silence
MIN_SPEECH_S = 0.4          # ignore segments shorter than this (avoids clangs/clicks)
ENERGY_THRESHOLD = 0.012    # simple energy VAD threshold, tune this for your room
MAX_SEGMENT_S = 15          # hard cap per utterance


def list_devices():
    print(sd.query_devices())


def energy_vad(frame: np.ndarray) -> bool:
    """Very simple RMS-energy VAD. Good enough for a first pass; swap for
    webrtcvad or silero-vad if ambient noise is triggering too many false
    positives."""
    rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
    return rms > ENERGY_THRESHOLD


def record_utterance(device, samplerate=MIC_SAMPLE_RATE) -> np.ndarray:
    """Blocks until it detects speech, records until silence, returns float32 mono audio."""
    frame_len = int(samplerate * FRAME_MS / 1000)
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[stream status] {status}", file=sys.stderr)
        q.put(indata.copy())

    print("Listening... (speak to trigger recording, Ctrl+C to quit)")
    buffer = []
    triggered = False
    silence_start = None
    start_time = None

    with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        blocksize=frame_len,
        device=device,
        callback=callback,
    ):
        while True:
            frame = q.get()[:, 0]
            is_speech = energy_vad(frame)

            if not triggered:
                if is_speech:
                    triggered = True
                    start_time = time.time()
                    silence_start = None
                    buffer = [frame]
                    print("  -> speech detected, recording...")
                continue

            buffer.append(frame)

            if is_speech:
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_TIMEOUT_S:
                    break

            if time.time() - start_time > MAX_SEGMENT_S:
                break

        audio = np.concatenate(buffer)
        duration = len(audio) / samplerate
        if duration < MIN_SPEECH_S:
            print(f"  -> segment too short ({duration:.2f}s), discarding")
            return np.array([], dtype=np.float32)
        print(f"  -> captured {duration:.2f}s")
        return audio


def denoise(audio_16k: np.ndarray) -> np.ndarray:
    """Upsample to 48k, run DeepFilterNet, downsample back to 16k for Whisper."""
    from df.enhance import enhance, init_df

    if not hasattr(denoise, "_model"):
        print("Loading DeepFilterNet model (first call only)...")
        denoise._model, denoise._df_state, _ = init_df()

    audio_48k = resample_poly(audio_16k, DF_SAMPLE_RATE, MIC_SAMPLE_RATE)
    import torch
    audio_tensor = torch.from_numpy(audio_48k).float().unsqueeze(0)
    enhanced = enhance(denoise._model, denoise._df_state, audio_tensor)
    enhanced_np = enhanced.squeeze(0).numpy()

    audio_16k_out = resample_poly(enhanced_np, WHISPER_SAMPLE_RATE, DF_SAMPLE_RATE)
    return audio_16k_out.astype(np.float32)


def get_speaker_model():
    """Loads the pretrained ECAPA-TDNN speaker embedding model (SpeechBrain).
    Cached on first call since loading is slow."""
    if not hasattr(get_speaker_model, "_model"):
        from speechbrain.inference.speaker import EncoderClassifier
        print("Loading ECAPA-TDNN speaker verification model (first call only)...")
        get_speaker_model._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )
    return get_speaker_model._model


def get_embedding(audio_16k: np.ndarray) -> np.ndarray:
    """Extracts a speaker embedding vector from 16kHz mono audio."""
    import torch
    model = get_speaker_model()
    signal = torch.from_numpy(audio_16k).float().unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_batch(signal)
    return embedding.squeeze().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def enroll(device, num_samples=4):
    """Records a few short utterances and stores the averaged embedding as
    your voiceprint. Re-run this if verification starts rejecting you too
    often (e.g. after a cold, or in a very different acoustic environment)."""
    print(f"Enrollment: you'll record {num_samples} short phrases.")
    print("Speak a normal sentence each time, e.g. 'Robot, this is my voice.'\n")
    embeddings = []
    for i in range(num_samples):
        input(f"Press Enter, then speak sample {i + 1}/{num_samples}...")
        audio = record_utterance(device)
        if audio.size == 0:
            print("  -> no speech captured, retrying this sample")
            continue
        processed = denoise(audio)
        embeddings.append(get_embedding(processed))

    if len(embeddings) < 2:
        print("Not enough valid samples captured. Run --enroll again.")
        return

    voiceprint = np.mean(embeddings, axis=0)
    np.save(VOICEPRINT_PATH, voiceprint)
    print(f"\nSaved voiceprint to {VOICEPRINT_PATH} (from {len(embeddings)} samples).")


def verify_speaker(audio_16k: np.ndarray, voiceprint: np.ndarray):
    """Returns (is_match, similarity) comparing this utterance to the enrolled voiceprint."""
    embedding = get_embedding(audio_16k)
    similarity = cosine_similarity(embedding, voiceprint)
    return similarity >= SPEAKER_SIMILARITY_THRESHOLD, similarity


def transcribe(audio_16k: np.ndarray, model):
    segments, info = model.transcribe(audio_16k, language=None, vad_filter=False)
    segments = list(segments)
    text = " ".join(s.text.strip() for s in segments)
    avg_logprob = (
        sum(s.avg_logprob for s in segments) / len(segments) if segments else float("-inf")
    )
    return text, avg_logprob, info.language, info.language_probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int, default=None, help="input device index (see --list-devices)")
    parser.add_argument("--no-denoise", action="store_true", help="skip DeepFilterNet, feed raw mic audio to Whisper (for A/B comparison)")
    parser.add_argument("--whisper-model", default="base", help="tiny/base/small/medium/large-v3")
    parser.add_argument("--confidence-threshold", type=float, default=-0.8, help="avg_logprob below this -> flagged low confidence")
    parser.add_argument("--enroll", action="store_true", help="record your voiceprint (run this once, first)")
    parser.add_argument("--no-speaker-check", action="store_true", help="disable speaker verification gate (debugging only)")
    parser.add_argument("--speaker-threshold", type=float, default=SPEAKER_SIMILARITY_THRESHOLD, help="cosine similarity cutoff for speaker match")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.device is None:
        print("Pass --device <index>. Run --list-devices first to find your microphone.")
        return

    if args.enroll:
        enroll(args.device)
        return

    voiceprint = None
    if not args.no_speaker_check:
        import os
        if not os.path.exists(VOICEPRINT_PATH):
            print(f"No voiceprint found at {VOICEPRINT_PATH}. Run with --enroll first,")
            print("or pass --no-speaker-check to skip speaker verification for now.")
            return
        voiceprint = np.load(VOICEPRINT_PATH)

    from faster_whisper import WhisperModel
    print(f"Loading Whisper model '{args.whisper_model}' (first call only)...")
    whisper_model = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")

    try:
        while True:
            audio = record_utterance(args.device)
            if audio.size == 0:
                continue

            if args.no_denoise:
                processed = audio
            else:
                processed = denoise(audio)

            # Gate 1: speaker verification -- reject anyone who isn't you
            if voiceprint is not None:
                is_match, similarity = verify_speaker(processed, voiceprint)
                if not is_match:
                    print("-" * 60)
                    print(f"REJECTED -- speaker not recognized (similarity {similarity:.2f} < {args.speaker_threshold:.2f})")
                    print("-" * 60)
                    continue
                print(f"  -> speaker verified (similarity {similarity:.2f})")

            # Gate 2: transcribe, then check confidence -- reject uncertain transcripts
            text, avg_logprob, lang, lang_prob = transcribe(processed, whisper_model)

            print("-" * 60)
            print(f"Transcript: {text!r}")
            print(f"Language: {lang} ({lang_prob:.2f})  |  avg_logprob: {avg_logprob:.3f}")
            if avg_logprob < args.confidence_threshold:
                print("LOW CONFIDENCE -> rejected. In a robot pipeline this triggers a re-ask, not an action.")
            else:
                print("ACCEPTED -> would be passed to NLP/intent parsing and robot action here.")
            print("-" * 60)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
