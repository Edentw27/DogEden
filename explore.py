"""
main.py  —  RoboDog full pipeline
==================================

    voice command  ->  parse colour+shape  ->  camera scan  ->  robot action

Runs in TWO modes automatically:
  * On your laptop (no xgolib)  -> SIMULATION: prints the action.
  * On the robot   (xgolib ok)  -> REAL: moves the dog via dog.action(id).

Vision: pure HSV colour + contour analysis (no trained model needed).
Voice:  Google Speech Recognition via the SpeechRecognition library.
"""

import time
from xgolib import XGO

dog = XGO('xgomini')
dog.reset()
time.sleep(1)

print("Action explorer. Type a number (1-20) to test a move, or 'q' to quit.")
print("Try: 1 get-down  2 stand  4 circle  6 squat  10 spin")
print("     12 sit  13 wave  14 stretch  16 swing  19 handshake  20 greet")

while True:
    s = input("action id> ").strip()
    if s.lower() == "q":
        break
    try:
        aid = int(s)
    except ValueError:
        print("  please type a number")
        continue
    try:
        dog.action(aid)
        time.sleep(5)      # let the move fully finish
        dog.reset()
        time.sleep(1)
    except Exception as e:
        print("  error:", e)

print("done")
