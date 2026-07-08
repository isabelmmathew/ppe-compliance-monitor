import cv2
import numpy as np

def draw_box(frame, pt1, pt2, colour, label=None):
    cv2.rectangle(frame, pt1, pt2, colour, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        lx, ly = pt1[0], pt1[1] - 6
        right = max(pt2[0], lx + tw + 4)
        cv2.rectangle(frame, (lx, ly - th - 4), (right, pt1[1]), colour, -1)
        cv2.putText(frame, label, (lx + 2, ly - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        
PADDING = 6
LINE_SPACING = 20

def draw_hud(frame, total, compliant, violations):
    lines = [
        f"Workers: {total}",
        f"Compliant: {compliant}",
        f"Violations: {violations}",
    ]
    y = LINE_SPACING
    max_tw = 0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        if(tw>max_tw): 
            max_tw = tw
    cv2.rectangle(frame, (0,0), (PADDING+max_tw+PADDING,len(lines)*LINE_SPACING+PADDING),(40,40,40),-1)
    for line in lines:
        cv2.putText(frame, line, (PADDING, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y += LINE_SPACING
    

def stack_images(scale, imgs):
    resized = []
    h, w = imgs[0].shape[:2]
    for img in imgs:
        img = cv2.resize(img, (w, h))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized.append(img)
    combined = np.hstack(resized)
    return cv2.resize(combined, (0,0), fx=scale, fy=scale)