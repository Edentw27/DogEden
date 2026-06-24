"""
tune_blue.py
-------------
Live visual tuner to separate the BLUE BALL from the BLUE CUBE.

Opens a camera window. Hold the blue object inside the green box. It outlines
the object and shows its measurements on screen AND prints them to the terminal:

    area  = size in pixels
    circ  = circularity (round things ~0.85, square things ~0.78)
    asp   = aspect ratio of the bounding box (both ~1.0 — not useful)
    ext   = extent: area / box area  (CIRCLE ~0.79, SQUARE ~0.95+)  <-- best bet
    v     = corner count (CUBE ~4, BALL more/rounded)               <-- also good

Do this:
  1. Hold the blue BALL in the box for ~5 seconds. Note ext and v.
  2. Hold the blue CUBE in the box for ~5 seconds. Note ext and v.
  3. Tell me both sets of numbers — I'll lock the threshold on whichever
     metric separates them cleanly.

Press q (with the window focused) to quit.
"""

import cv2
import numpy as np
import time

BLUE_LOW  = np.array([100, 50, 50])
BLUE_HIGH = np.array([130, 255, 255])
MIN_AREA = 5000

cap = cv2.VideoCapture(0)
last_print = time.time()

print("Hold the blue object in the green box. Watch the on-screen numbers.")
print("Hold the BALL ~5s, then the CUBE ~5s. Press q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    x0, y0, x1, y1 = int(w * 0.30), int(h * 0.20), int(w * 0.70), int(h * 0.80)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

    roi = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5),   np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label = "no blue object"
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > MIN_AREA:
            peri = cv2.arcLength(c, True)
            circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 1
            extent = area / (bw * bh) if bw * bh > 0 else 0
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            verts = len(approx)

            cv2.drawContours(roi, [c], -1, (0, 0, 255), 2)
            label = f"area={area:.0f} circ={circ:.2f} asp={aspect:.2f} ext={extent:.2f} v={verts}"

            if time.time() - last_print > 0.5:
                last_print = time.time()
                print("  " + label)

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("blue tuner - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()