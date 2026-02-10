import cv2
import matplotlib.pyplot as plt
import numpy as np


# 1. Loading to Image 
FILENAME = 'images/WIN_20260202_10_31_38_Pro.jpg'

# 2. Threshold Value
THRESH_VAL = 100 

# 3. Minimum Area Filter

MIN_AREA = 1000 

# --- PIPELINE ---

# 1. Load Image
img = cv2.imread(FILENAME)

if img is None:
    print(f"Error: Image '{FILENAME}' not found. Check path.")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Thresholding (Inverse)
    # background is light, objects are dark -> Inverse makes objects white.
    _, binary = cv2.threshold(gray, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)

    # 2. Morphological cleanup
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3. Connected Components 
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8, ltype=cv2.CV_32S
    )

    print(f"Total blobs found (before filtering): {num_labels - 1}")

    # Create a copy to draw on
    output_img = img.copy()

    # Counter for valid objects
    valid_object_count = 0

    # Loop starting from 1 to ignore background
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        cx, cy = int(centroids[label][0]), int(centroids[label][1])

        # --- NEW STEP: AREA FILTER ---
        # Only process if the blob is big enough
        if area > MIN_AREA:
            valid_object_count += 1
            
            # --- VISUALIZATION ---
            # 1. Draw Box (MAGENTA)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 255), 5) 
            
            # 2. Draw Centroid (RED DOT)
            cv2.circle(output_img, (cx, cy), 15, (0, 0, 255), -1) 
            
            # 3. Put Text Label (GREEN COORDINATES)
            text_label = f"({cx},{cy})"
            text_pos = (cx - 60, cy - 20) 
            cv2.putText(output_img, text_label, text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            print(f" Object {label} accepted: Area={area}, Pos=({cx},{cy})")

    print(f"Final valid objects detected: {valid_object_count}")

    # --- DISPLAY ---
    # Convert BGR to RGB for Matplotlib
    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    
    # Also verify the binary mask to see what the computer "sees"
    binary_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)

    # Show both images side-by-side
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(binary_rgb)
    plt.title("Step 1: Binary Mask (What computer sees)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_rgb)
    plt.title("Step 2: Final Result")
    plt.axis('off')
    
    plt.show()