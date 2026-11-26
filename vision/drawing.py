"""Common OpenCV text and panel headings."""

import cv2


def draw_drive_mode(image, mode):
    label = "Mode: " + mode
    # Keep the label legible when the three-panel debug view is scaled down.
    scale = max(.5, image.shape[1] / 1700)
    thickness = max(1, round(scale * 2))
    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(0, image.shape[1] - w - 14)
    cv2.rectangle(image, (x, 0), (image.shape[1]-1, h+baseline+12), (20, 20, 20), -1)
    draw_text(image, label, (x+6, h+5), scale, thickness=thickness)


def draw_text(image, text, position, scale=.5, colour=(255, 255, 255), thickness=1):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness)


def title_panel(panel, title):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    draw_text(panel, title, (8, 22))
    return panel
