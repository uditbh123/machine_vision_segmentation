import cv2
import matplotlib.pyplot as plt

img = cv2.imread('images/image_2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(binary, cmap='gray')
plt.title('Binary after threshold'); plt.axis('off')
plt.show()
