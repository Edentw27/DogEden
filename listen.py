"""
listen.py
----------
Standalone voice-command tester. Listens once, prints what it heard and
the parsed object name. Use this to check your microphone and the speech
recognition before running the full main.py.

Say something like: "fetch green ball"
"""

import speech_recognition as sr

VALID_COLORS = ["red", "green", "blue", "yellow", "purple"]
VALID_SHAPES = ["ball", "cube"]


def parse_command(text):
    text = text.lower()
    color = next((c for c in VALID_COLORS if c in text), None)
    shape = next((s for s in VALID_SHAPES if s in text), None)
    if color and shape:
        return f"{color}_{shape}"
    return None


def listen_once():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening... say e.g. 'fetch green ball'")
        audio = r.listen(source, phrase_time_limit=4)
    try:
        text = r.recognize_google(audio)
        print(f"Heard: \"{text}\"")
        target = parse_command(text)
        if target:
            print(f"Parsed target: {target}")
        else:
            print("Couldn't find a colour + shape in that. Try again.")
    except sr.UnknownValueError:
        print("Couldn't understand the audio.")
    except sr.RequestError as e:
        print(f"Speech service error: {e}")


if __name__ == "__main__":
    while True:
        listen_once()
        again = input("\nListen again? (y/n) ").strip().lower()
        if again != "y":
            break
