import cv2
import matplotlib.pyplot as plt

# step 1 load image and convert to binary 
img = cv2.imread('images/image_2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# step 2 morhological operation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Opening: erosion → dilation  (removes noise specks)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# Closing: dilation → erosion  (fills holes inside objects)
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

# show before and after 
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1); plt.imshow(binary, cmap='gray')
plt.title('After threshold (step 1)'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(cleaned, cmap='gray')
plt.title('After morphological operation (Step 2)'); plt.axis('off')
plt.show()
