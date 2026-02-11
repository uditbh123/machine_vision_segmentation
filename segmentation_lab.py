import cv2
import matplotlib.pyplot as plt
import numpy as np

def analyze_image(filename):
    print(f"\nProcessing: {filename}")
    img = cv2.imread(filename)
    if img is None: 
        print(f"Error: Could not load {filename}")
        return

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. HEAVY BLUR (The Fix)
    # We increase blur from (5,5) to (15,15). 
    # This smears the edges into the center, making the hollow tile solid.
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # 3. Adaptive Thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        101,  # Large block size to handle shadows
        2     # Low C to be very sensitive to faint objects
    )

    # 4. Morphology (Standard cleanup)
    # We don't need huge kernels anymore because the blur did the heavy lifting.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Open to remove noise
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # Dilate slightly to smooth edges
    dilated = cv2.dilate(opened, kernel, iterations=2)
    # Close to fill any remaining small pinholes
    cleaned = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5. Connected Components
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8, ltype=cv2.CV_32S
    )

    # 6. Annotation Loop
    output_img = img.copy()
    valid_count = 0
    
    # Dynamic Area Filter
    h, w = img.shape[:2]
    min_area = (h * w) * 0.001 
    
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        cx, cy = int(centroids[label][0]), int(centroids[label][1])

        if area > min_area:
            valid_count += 1
            # Draw Box (Magenta)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 255), 5)
            # Draw Centroid (Red)
            cv2.circle(output_img, (cx, cy), 15, (0, 0, 255), -1)
            # Draw Text (Green)
            cv2.putText(output_img, f"({cx},{cy})", (cx - 60, cy - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    print(f"  > Final valid objects: {valid_count}")

    # 7. Display
    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    binary_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(binary_rgb)
    plt.title("Binary Mask (Heavy Blur Applied)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_rgb)
    plt.title(f"Found {valid_count} Objects")
    plt.axis('off')
    
    plt.show()
# --- MAIN BLOCK ---
if __name__ == "__main__":
    # 1. Test Real Image
    analyze_image('images/manyobjects.jpg')
    
    # 2. Test Simulated Image
    analyze_image('images/image_2.png')