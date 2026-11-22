"""Common OpenCV text and panel headings."""

import cv2


def draw_text(image, text, position, scale=.5, colour=(255, 255, 255), thickness=1):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness)


def title_panel(panel, title):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    draw_text(panel, title, (8, 22))
    return panel
