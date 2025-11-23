import cv2
import numpy as np
import pandas as pd

def identify_stable_burning(image_paths, threshold_area, group_size):
    # Create DataFrames for datasets
    dataset1 = pd.DataFrame(columns=['Image', 'Label'])  # Dataset for stable burning
    dataset2 = pd.DataFrame(columns=['Image', 'Label'])  # Dataset for no stable burning

    # Counter for grouping images
    counter = 0

    for image_path in image_paths:
        # Load image
        img = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold image
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area or other criteria
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > threshold_area]

        # Draw contours on the original image
        result = img.copy()
        cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

        # Classify and save the image in the respective dataset
        if len(filtered_contours) > 1:
            print(f"Stable burning detected in {image_path}")
            dataset1.loc[len(dataset1)] = [image_path, "Stable Burning"]
        else:
            print(f"No stable burning detected in {image_path}")
            dataset2.loc[len(dataset2)] = [image_path, "No Stable Burning"]

        # Increment counter
        counter += 1

        # Display the images in groups
        if counter % group_size == 0 or counter == len(image_paths):
            cv2.imshow('Stable Burning', result)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    dataset1.to_csv(r'D:\Users\91805\Documents\Punit Docs\semster 2\project image processsing\dataset1.csv', index=False)
    dataset2.to_csv(r'D:\Users\91805\Documents\Punit Docs\semster 2\project image processsing\dataset2.csv', index=False)
 # Set the list of image paths
image_paths = [
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\burnedfield1.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\burnedfield2.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\burnedfield3.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\burnedfield4.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\burnedfield5.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\burnedfield6.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\burnedfield7.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\burnedfield8.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\field.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\field1.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\field2.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\field3.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\field4.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\field5.jpg",
    "D:\\Users\\91805\\Documents\\Punit Docs\\semster 2\\project image processsing\\field6.jpg",
]

# Set the threshold area for filtering contours
threshold_area = 1000

# Set the group size for displaying images
group_size = 3

# Call the function to identify stable burning and display images in groups
identify_stable_burning(image_paths, threshold_area, group_size)
dataset1 = pd.read_csv(r'D:\Users\91805\Documents\Punit Docs\semster 2\project image processsing\dataset1.csv')
dataset2 = pd.read_csv(r'D:\Users\91805\Documents\Punit Docs\semster 2\project image processsing\dataset2.csv')

# Access and print the data in dataset1
print("Dataset 1:")
print(dataset1.head())

# Access and print the data in dataset2
print("Dataset 2:")
print(dataset2.head())
