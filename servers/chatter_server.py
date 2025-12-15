import os
import json
import asyncio
import websockets
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from pydub.playback import play
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Configuration
DEFAULT_AUDIO_PROMPT = "./cb_2.wav"

print("Loading Chatterbox TTS model...")
model = ChatterboxTTS.from_pretrained(device="cuda")
print("Model loaded successfully")

# Thread pool for non-blocking audio playback
executor = ThreadPoolExecutor(max_workers=2)

def play_audio_async(audio_path):
    """Non-blocking audio playback"""
    try:
        audio = AudioSegment.from_wav(audio_path)
        play(audio)
    except Exception as e:
        print(f"Audio playback error: {e}")

async def tts_server(websocket):
    """Handle TTS requests via WebSocket"""
    try:
        while True:
            # Receive request
            packet = json.loads(await websocket.recv())
            text = packet.get("text", "")
            audio_prompt_path = packet.get("audio_prompt_path", DEFAULT_AUDIO_PROMPT)
            save_path = f"""../../runtime/speech/{packet.get("save_path", "temp_speech.wav")}"""
            play_audio = packet.get("play_audio", True)

            # Validate inputs
            if not text:
                await websocket.send(json.dumps({"error": "No text provided"}))
                continue

            # Check if audio prompt exists
            if not os.path.exists(audio_prompt_path):
                await websocket.send(json.dumps({
                    "error": f"Audio prompt not found: {audio_prompt_path}",
                    "default_prompt": DEFAULT_AUDIO_PROMPT
                }))
                continue

            # Generate audio
            try:
                print(f"Generating speech: {text[:50]}...")
                wav = model.generate(text, audio_prompt_path=audio_prompt_path)

                # Save audio file
                sf.write(
                    save_path,
                    wav.cpu().numpy().squeeze(),
                    model.sr,
                    subtype='PCM_16'
                )

                print(f"Audio saved to: {save_path}")

                # Optional: Play audio asynchronously (non-blocking)
                if play_audio:
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(executor, play_audio_async, save_path)

                await websocket.send(json.dumps({
                    "status": "success",
                    "message": "TTS processing complete",
                    "output_path": save_path
                }))

            except Exception as e:
                print(f"Inference error: {str(e)}")
                await websocket.send(json.dumps({
                    "error": f"Inference failed: {str(e)}"
                }))

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f"Server error: {e}")

# Start server
async def main():
    print("Starting Chatterbox TTS server on ws://localhost:8767")
    print(f"Default audio prompt: {DEFAULT_AUDIO_PROMPT}")
    async with websockets.serve(tts_server, "localhost", 8767):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
