import asyncio
import websockets
import json
import argparse

async def send_tts_request(text, audio_prompt_path="./hydrogen.wav", save_path="output.wav",
                          play_audio=True, host="localhost", port=8767, verbose=True):
    """Send TTS request to Chatterbox server"""
    uri = f"ws://{host}:{port}"

    try:
        async with websockets.connect(uri) as websocket:
            request = {
                "text": text,
                "audio_prompt_path": audio_prompt_path,
                "save_path": save_path,
                "play_audio": play_audio
            }

            if verbose:
                print(f"Sending request to {uri}...")
            await websocket.send(json.dumps(request))

            response = json.loads(await websocket.recv())

            if "error" in response:
                if verbose:
                    print(f"Error: {response['error']}")
                    if "available_styles" in response:
                        print(f"Available styles: {response['available_styles']}")
                return False
            else:
                if verbose:
                    print(f"Status: {response['status']}")
                    print(f"Message: {response['message']}")
                    print(f"Output: {response['output_path']}")
                return True

    except (ConnectionRefusedError, OSError) as e:
        if verbose:
            print(f"Error: Could not connect to server at {uri}")
            print("Make sure the server is running with: python chatter_server.py")
        return False
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        return False

def send_tts_sync(text, audio_prompt_path="./space_startup.wav", save_path="output.wav",
                 play_audio=True, host="localhost", port=8767, verbose=True):
    """Synchronous wrapper for send_tts_request"""
    return asyncio.run(send_tts_request(
        text=text,
        audio_prompt_path=audio_prompt_path,
        save_path=save_path,
        play_audio=play_audio,
        host=host,
        port=port,
        verbose=verbose
    ))

def main():
    parser = argparse.ArgumentParser(description="Chatterbox TTS Client")

    parser.add_argument(
        "text",
        type=str,
        help="Text to synthesize"
    )

    parser.add_argument(
        "--audio-prompt",
        type=str,
        default="./nonfiction_voice.wav",
        # default="./unify_7.wav",
        # default="./hydrogen.wav",
        help="Path to audio prompt file (default: ./space_drama.wav)"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output file path (default: output.wav)"
    )

    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Don't play audio after generation"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8767,
        help="Server port (default: 8767)"
    )

    args = parser.parse_args()

    asyncio.run(send_tts_request(
        text=args.text,
        audio_prompt_path=args.audio_prompt,
        save_path=args.output,
        play_audio=not args.no_play,
        host=args.host,
        port=args.port
    ))

if __name__ == "__main__":
    main()
