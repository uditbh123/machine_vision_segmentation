import cv2
import matplotlib.pyplot as plt
import numpy as np

def analyze_image(filename, min_area, use_adaptive=False, global_thresh=100):
    
    print(f"\nProcessing: {filename}")
    img = cv2.imread(filename)
    if img is None: return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- STEP 1: THRESHOLDING ---
    if use_adaptive:
        print("  > Using Adaptive Thresholding (Real Image Mode)")
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 51, 5
        )
    else:
        print(f"  > Using Global Thresholding (Simulated Mode): {global_thresh}")
        _, binary = cv2.threshold(gray, global_thresh, 255, cv2.THRESH_BINARY_INV)

    # --- STEP 2: MORPHOLOGICAL CLEANUP (Improved) ---
    
    # 1. Remove noise (small specks)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # 2. Thicken the lines (Dilation) -- NEW STEP
    # This expands the white lines to bridge any gaps in the tile edges.
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(opened, kernel_medium, iterations=2)
    
    # 3. Fill the centers (Closing)
    # Now that the outline is solid and thick, we fill the inside.
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)) 
    cleaned = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_large, iterations=1)

    # --- STEP 3: CONNECTED COMPONENTS ---
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8, ltype=cv2.CV_32S
    )

    # --- STEP 4: FILTER & DRAW ---
    output_img = img.copy()
    valid_count = 0

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        cx, cy = int(centroids[label][0]), int(centroids[label][1])

        if area > min_area:
            valid_count += 1
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 255), 5)
            cv2.circle(output_img, (cx, cy), 15, (0, 0, 255), -1)
            cv2.putText(output_img, f"({cx},{cy})", (cx - 60, cy - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # --- STEP 5: DISPLAY ---
    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    binary_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1); plt.imshow(binary_rgb); plt.title("Binary Mask"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(output_rgb); plt.title(f"Found {valid_count} Objects"); plt.axis('off')
    plt.show()

# --- HOW TO RUN ---
if __name__ == "__main__":
    
    # 1. For SIMULATION (Perfect Light) -> Use Global
    # analyze_image('images/image_2.png', min_area=100, use_adaptive=False, global_thresh=200)

    # 2. For REAL IMAGE (Uneven Light) -> Use Adaptive
    analyze_image('images/WIN_20260202_10_32_43_Pro.jpg', min_area=1000, use_adaptive=True)