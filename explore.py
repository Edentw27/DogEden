"""
explore.py  —  RUN THIS ON THE ROBOT
=====================================
Helps you discover which xgolib preset action id does what, so you can fill in
ACTION_IDS in main.py correctly.

There is NO dog.sit() in this xgolib version — preset moves go through
dog.action(id) where id is 1..255, and dog.reset() returns to neutral.

Put the robot on a clear, flat surface with space around it before running.
Type an id, watch what it does, note it down. 'q' to quit.
"""

import time
from xgolib import XGO

dog = XGO('xgomini')
dog.reset()
time.sleep(1)

print("Robot action explorer. Type an id 1-255, or 'q' to quit.")
print("Suggested ids to try: 1 2 3 4 6 11 12 17 19 21 24")

while True:
    s = input("action id> ").strip()
    if s.lower() == "q":
        break
    try:
        aid = int(s)
        dog.action(aid, wait=True)
    except ValueError:
        print("  enter a number")
        continue
    except Exception as e:
        print("  error:", e)
    time.sleep(1)
    dog.reset()

print("done")
