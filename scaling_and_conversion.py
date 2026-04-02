# This program performs image preprocessing:
# - Converts images to 8-bit grayscale
# - Resizes images to a fixed dimension (optional)
# Note: Images should be organized in directories corresponding to class labels (e.g., 0, 1, 2)

print("\033[2J")  # Clears the console (optional, for cleaner output)

import cv2
import glob, os

# *** SETTINGS ***
path = '1'                  # Directory containing images for a specific class
newDimension = (200, 200)   # Target image size (width, height)
resizing = 1                # 1 = enable resizing, 0 = disable resizing

def main():
    # Retrieve list of all image file paths in the specified directory
    fileList = glob.glob(path + '/*')    
    fileIndex = 0

    for filePath in fileList:
        fileIndex += 1
        fileNumber = len(fileList)  # Total number of images

        # Extract original file name (not used further, but kept for reference)
        _, fileName = os.path.split(filePath)

        # Generate new file name in BMP format (sequential numbering)
        fileName = str(fileIndex) + '.bmp'

        # Read image in grayscale mode (8-bit)
        image = cv2.imread(filePath, 0)  # 0 -> grayscale

        # Resize image if resizing is enabled
        if resizing:
            try:
                image = cv2.resize(image, newDimension)
            except:
                # Stop processing if resizing fails (e.g., corrupted image)
                break

        # Save processed image in the same directory with new name
        if image is not None:
            cv2.imwrite(path + '/' + fileName, image)

        # Remove the original image file to avoid duplication
        os.remove(filePath)

        # Print processing progress
        print(str(fileIndex) + '/' + str(fileNumber) + ' - ' + fileName + ' - PROCESSED')


if __name__ == '__main__':
    main()