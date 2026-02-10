import cv2
import matplotlib.pyplot as plt

# step 1 load, grayscale, threshold
img = cv2.imread('images/image_2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

#STEP 2: Morphological cleanup

kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

# step 3 connected componets 
num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
    cleaned, connectivity=8, ltype=cv2.CV_32S
)

print(f"Found {num_labels - 1} objects")

for label in range(1, num_labels):

    x, y, w, h, area = stats[label]

    cx, cy = int(centroids[label][0]), int(centroids[label][1])

    print(f"  Objects {label}:")
    print(f"    Area   = {area} pixels")
    print(f"    Centroid = ({cx}, {cy})")
    print(f"    Box      = top-left({x},{y})  size {w}x{h}")
    