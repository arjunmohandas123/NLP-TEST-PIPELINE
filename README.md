# NLP Test Pipeline — Voice Command Front-End

A small, self-contained test harness for the **speech front-end of a voice-controlled robot / NLP system**. It validates the full software chain from microphone to a clean, trusted transcript:

```
mic capture → energy VAD → DeepFilterNet denoise → speaker-verification gate → Whisper ASR → confidence gate → (your NLP / intent logic)
```

Two independent gates decide whether a spoken command is allowed through to downstream NLP or robot actions:

1. **Speaker verification** — is this the enrolled user's voice? Anyone else is rejected. Uses a pretrained ECAPA-TDNN embedding model (SpeechBrain) and cosine similarity against a locally stored voiceprint.
2. **Confidence gate** — is the transcript reliable? Whisper's average log-probability must exceed a threshold; otherwise the command is flagged for a "please repeat" instead of an action.

Everything runs on CPU. The goal is to test *pipeline logic* (gating, thresholds, A/B with and without denoising), not to benchmark acoustic robustness.

---

## Repository contents

| File | Purpose |
|---|---|
| `audio_pipeline.py` | The complete pipeline: device listing, VAD recording, denoising, speaker enrollment/verification, transcription, gating. |
| `requirements.txt` | Python dependencies. |
| `.gitignore` | Keeps voiceprints, recordings, downloaded model weights and virtual environments out of the repository. |

> **Privacy note:** no audio recordings, voiceprints or model weights are stored in this repository. Enrollment creates a `voiceprint.npy` locally and the models download into `pretrained_models/` on first run; both are git-ignored and stay on your machine.

---

## Setup

Python 3.10–3.12 is recommended.

```bash
git clone <repo-url>
cd nlp-test-pipeline
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Notes:

* `deepfilternet`, `speechbrain` and `torchaudio` pull in PyTorch — the CPU build is sufficient and no GPU is required, but it is a large download.
* `sounddevice` needs PortAudio. On Debian/Ubuntu: `sudo apt install libportaudio2`. macOS and Windows wheels bundle it.
* On first run the ECAPA-TDNN speaker model (~80 MB) and the chosen Whisper model are downloaded automatically.

---

## Usage

**1. Find your microphone's device index**

```bash
python audio_pipeline.py --list-devices
```

**2. Enroll your voice (one time)**

Records four short phrases and stores the averaged speaker embedding as `voiceprint.npy`.

```bash
python audio_pipeline.py --device 3 --enroll
```

Re-run enrollment if verification starts rejecting you too often (different room, different headset, a cold…).

**3. Run the pipeline**

```bash
python audio_pipeline.py --device 3
```

Speak a command. The script detects speech, records until you go quiet, denoises, checks the speaker, transcribes, and prints the transcript with its confidence score and the gate decision (`ACCEPTED`, `REJECTED -- speaker not recognized`, or `LOW CONFIDENCE`).

**Useful variants**

```bash
python audio_pipeline.py --device 3 --no-denoise          # A/B: raw mic audio straight to Whisper
python audio_pipeline.py --device 3 --no-speaker-check    # debugging: bypass the speaker gate
python audio_pipeline.py --device 3 --whisper-model small # tiny / base / small / medium / large-v3
python audio_pipeline.py --device 3 --confidence-threshold -0.6 --speaker-threshold 0.6
```

---

## How it works

| Stage | Implementation | Where to change it |
|---|---|---|
| Capture | `sounddevice` input stream, 16 kHz mono, 30 ms frames | `MIC_SAMPLE_RATE`, `FRAME_MS` |
| Voice activity detection | RMS-energy threshold; recording stops after 1 s of silence, min 0.4 s, max 15 s per utterance | `ENERGY_THRESHOLD`, `SILENCE_TIMEOUT_S`, `MIN_SPEECH_S`, `MAX_SEGMENT_S` |
| Denoising | Resample 16 k → 48 k, DeepFilterNet `enhance()`, resample back to 16 k | `denoise()` |
| Speaker gate | SpeechBrain `spkrec-ecapa-voxceleb` embeddings, cosine similarity ≥ 0.55 | `SPEAKER_SIMILARITY_THRESHOLD` / `--speaker-threshold` |
| ASR | `faster-whisper` (CTranslate2, int8 on CPU), language auto-detect | `--whisper-model` |
| Confidence gate | Mean `avg_logprob` over segments ≥ −0.8 | `--confidence-threshold` |

Models are loaded lazily and cached on first use, so the first utterance is slow and subsequent ones are fast.

---

## Tuning tips

* **`ENERGY_THRESHOLD`** is the crudest part of the pipeline. Raise it if ambient noise (tools, doors, machinery) falsely triggers "speech detected"; lower it if quiet speech is missed. For anything beyond initial testing, swap in `webrtcvad` or `silero-vad`.
* **Whisper model size:** start with `base` for speed; move to `small` or `medium` when measuring real accuracy under noise.
* **Confidence threshold:** log a batch of real transcripts against ground truth and pick the cutoff that separates correct from garbled commands in *your* environment rather than trusting the default.
* **Speaker threshold:** 0.55 is a reasonable starting point for ECAPA-TDNN; raise it for stricter rejection of other speakers, lower it if the enrolled user is being rejected.

---

## Known limitations

* Bluetooth headset microphones use a narrow-band call profile, which caps ASR accuracy below a wired/USB mic or a robot-mounted array.
* The energy VAD misreads sudden loud transients as speech; expect some false triggers in a noisy room at this stage.
* Single channel only — no beamforming or multi-mic processing.
* Speaker verification on short, noisy utterances is probabilistic; it is a gate, not an authentication mechanism.

---

## Next steps

Plug the `ACCEPTED` transcript into an intent parser / NLP module, replace the energy VAD with a learned one, and add structured logging of (transcript, similarity, avg_logprob, decision) to calibrate both thresholds on real data.
