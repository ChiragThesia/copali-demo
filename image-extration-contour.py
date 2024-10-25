import cv2
import numpy as np

# Load image
image = cv2.imread('test1.png')  # Change to your actual image file path
if image is None:
    print("Error: Could not read the image. Check the file path and integrity.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's threshold to binarize the image (inverse binary to target non-text areas)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Dilate with a slightly smaller kernel to avoid merging too much
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))  # Reduced kernel size
dilate = cv2.dilate(thresh, kernel, iterations=2)

# Find contours in the image
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Image dimensions
img_height, img_width = image.shape[:2]

# Initialize variables to store cropped areas
x_min, y_min = img_width, img_height
x_max, y_max = 0, 0

# Iterate through contours and filter out text-like regions
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)

    # Filter out large areas likely to be text (based on size and aspect ratio)
    if area > 3000:  # Lowered threshold to prevent over-removing
        aspect_ratio = w / float(h)

        # Skip text regions based on wide aspect ratio or regions near the image boundaries
        if aspect_ratio > 5 or h < 40 or w > img_width * 0.9 or y < img_height * 0.05 or y + h > img_height * 0.95:
            # Optionally, draw green rectangles around text regions (visualization)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue  # Skip text-like areas
        else:
            # Update the bounding box to include non-text areas
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

# Crop the remaining non-text area
cropped_image = None
if x_min < x_max and y_min < y_max:
    cropped_image = image[y_min:y_max, x_min:x_max]
    cv2.imwrite('diagram_no_text_refined.png', cropped_image)  # Save the cropped image

    # Draw a red rectangle around the cropped diagram for visualization
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

# Display the original image with rectangles and the cropped diagram
cv2.imshow('image', image)
if cropped_image is not None:
    cv2.imshow('Cropped Diagram', cropped_image)
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
