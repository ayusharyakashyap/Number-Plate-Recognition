import cv2
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as plt

# Load image and convert to grayscale
img = cv2.imread('image4.jpg')

if img is None:
    print("⚠️ Error: Image not found! Check the file path.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.title("Grayscale Image")
plt.show()

# Noise reduction and edge detection
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise Reduction
edged = cv2.Canny(bfilter, 30, 200)  # Edge Detection

plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.title("Edge Detection")
plt.show()

# Find contours
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Debug: Draw detected contours
debug_img = img.copy()
cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
plt.title("All Detected Contours")
plt.show()

# Find 4-point contour (likely license plate)
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        location = approx.astype(int)
        break

if location is None:
    print("⚠️ License plate not detected! Try adjusting edge detection parameters.")
    exit()

# Create mask and extract license plate
mask = np.zeros(gray.shape, dtype=np.uint8)
cv2.drawContours(mask, [location], 0, 255, -1)

new_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title("Extracted License Plate")
plt.show()

# Crop the license plate region
(x, y) = np.where(mask == 255)
if len(x) == 0 or len(y) == 0:
    print("⚠️ Error: Unable to crop the license plate.")
    exit()

(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title("Cropped License Plate")
plt.show()

# OCR for text detection
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)

if not result:
    print("⚠️ No text detected!")
    exit()

# Extract all detected text
detected_texts = [text[-2] for text in result]
text = " ".join(detected_texts)  # Combine all detected text if multiple words

# Overlay text on the original image
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(
    img, 
    text=text, 
    org=(location[0][0][0], location[1][0][1] + 60), 
    fontFace=font, 
    fontScale=1, 
    color=(0, 255, 0), 
    thickness=2, 
    lineType=cv2.LINE_AA
)

res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title("Final Image with License Plate Text")
plt.show()