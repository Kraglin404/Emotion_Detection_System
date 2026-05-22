import cv2
import numpy as np

# Actual evaluation results
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
confusion = np.array([
    [399, 2, 26, 10, 322, 85, 114],
    [38, 14, 1, 2, 40, 9, 7],
    [85, 0, 133, 10, 454, 245, 97],
    [62, 2, 13, 1234, 137, 117, 209],
    [53, 0, 9, 16, 897, 60, 212],
    [9, 0, 14, 15, 54, 722, 17],
    [17, 2, 7, 30, 398, 58, 721]
], dtype=np.int32)

# Grid parameters for high quality
cell_size = 90
margin_left = 130
margin_top = 100
margin_right = 50
margin_bottom = 80

img_w = margin_left + len(EMOTIONS) * cell_size + margin_right
img_h = margin_top + len(EMOTIONS) * cell_size + margin_bottom

# Create a clean modern dark background (charcoal color: #1e1e24 -> BGR: [36, 30, 30])
img = np.full((img_h, img_w, 3), [36, 30, 30], dtype=np.uint8)

# Title and subtitle
cv2.putText(img, "Confusion Matrix - Facial Emotion Detection (FER-2013)", (margin_left - 30, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, "Values show absolute counts and recall % (normalized by true class size)", (margin_left - 30, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

# Draw column headers (Predicted)
cv2.putText(img, "Predicted Class", (margin_left + 220, margin_top - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)

for col_idx, emo in enumerate(EMOTIONS):
    x = margin_left + col_idx * cell_size + (cell_size // 2) - 15
    y = margin_top - 15
    cv2.putText(img, emo[:3], (x - 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

# Draw row headers (True Class)
for i, char in enumerate("TRUE CLASS"):
    cv2.putText(img, char, (25, margin_top + 160 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)

for row_idx, emo in enumerate(EMOTIONS):
    x = margin_left - 100
    y = margin_top + row_idx * cell_size + (cell_size // 2) + 5
    cv2.putText(img, emo, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

# Plot cells with smooth blue heatmap gradient
for r in range(len(EMOTIONS)):
    row_sum = np.sum(confusion[r, :])
    for c in range(len(EMOTIONS)):
        val = confusion[r, c]
        pct = (val / row_sum) if row_sum > 0 else 0.0
        
        # ROYAL BLUE GRADIENT CALCULATION
        # Low recall -> Darker base color
        # High recall -> Bright royal blue (BGR: [220, 120, 30])
        b = int(45 + pct * 185)
        g = int(35 + pct * 95)
        r_val = int(30 + pct * 15)
        
        cell_x1 = margin_left + c * cell_size
        cell_y1 = margin_top + r * cell_size
        cell_x2 = cell_x1 + cell_size
        cell_y2 = cell_y1 + cell_size
        
        # Fill cell with gradient color
        cv2.rectangle(img, (cell_x1, cell_y1), (cell_x2, cell_y2), (b, g, r_val), cv2.FILLED)
        # Outline cell with crisp border
        cv2.rectangle(img, (cell_x1, cell_y1), (cell_x2, cell_y2), (60, 55, 55), 1)
        
        text_count = str(val)
        text_pct = f"{pct:.1%}"
        
        # Automatically flip text color for high contrast
        text_color = (0, 0, 0) if pct > 0.45 else (235, 235, 235)
        
        # Position and draw cell texts
        cv2.putText(img, text_count, (cell_x1 + 22, cell_y1 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
        cv2.putText(img, text_pct, (cell_x1 + 16, cell_y1 + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA)

# Save image
cv2.imwrite("training/confusion_matrix.png", img)
print("Beautiful confusion matrix generated at training/confusion_matrix.png successfully!")
