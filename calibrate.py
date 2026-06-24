import cv2, numpy as np, time

COLOR_RANGES = {
    "red":    [([0, 150, 80], [6, 255, 255]), ([170, 150, 80], [180, 255, 255])],
    "green":  [([35, 60, 50],   [85, 255, 255])],
    "blue":   [([100, 50, 50],  [130, 255, 255])],
    "purple": [([140, 50, 100], [170, 255, 255])],
    "yellow": [([18, 120, 120], [32, 255, 255])],
}
MIN_AREA, MAX_AREA = 8000, 250000

def get_mask(hsv, ranges):
    c = None
    for lo, up in ranges:
        m = cv2.inRange(hsv, np.array(lo), np.array(up))
        c = m if c is None else cv2.bitwise_or(c, m)
    return c

cap = cv2.VideoCapture(0)
last = time.time()
print("Hold ONE object in the CENTER. Ctrl+C to quit.\n")
try:
    while True:
        ret, frame = cap.read()
        if not ret: continue
        if time.time() - last < 0.5: continue
        last = time.time()
        h, w = frame.shape[:2]
        roi = frame[int(h*0.20):int(h*0.80), int(w*0.30):int(w*0.70)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        rh, rw = roi.shape[:2]
        patch = hsv[rh//2-30:rh//2+30, rw//2-30:rw//2+30]
        line = f"center HSV: H={patch[:,:,0].mean():4.0f} S={patch[:,:,1].mean():4.0f} V={patch[:,:,2].mean():4.0f}"
        for name, ranges in COLOR_RANGES.items():
            mask = get_mask(hsv, ranges)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15,15),np.uint8))
            cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                big = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(big)
                if MIN_AREA <= area <= MAX_AREA:
                    peri = cv2.arcLength(big, True)
                    circ = 4*np.pi*area/(peri*peri) if peri>0 else 0
                    x,y,bw,bh = cv2.boundingRect(big)
                    asp = min(bw,bh)/max(bw,bh) if max(bw,bh)>0 else 1
                    line += f" | {name}: area={area:6.0f} circ={circ:.2f} asp={asp:.2f}"
        print(line)
except KeyboardInterrupt:
    pass
finally:
    cap.release(); print("\nstopped")
