# License Plate Detection and OCR

This repository contains a project for **License Plate Detection and Optical Character Recognition (OCR)** using OpenCV and EasyOCR. The project extracts license plate text from images using edge detection, contour approximation, and text recognition.

## 📂 Folder Structure

```
📦 Project Root
│-- 📂 Main Code
│   │-- NPD.ipynb          # Final Jupyter Notebook implementation
│   │-- NPD.py             # Final Python script implementation
│   │-- image1.jpg         # Test images for the final code
│   │-- image2.jpg         # Test images for the final code
│   │-- image3.jpg         # Test images for the final code
│   │-- image4.jpg         # Test images for the final code
│   │-- yolov8n.pt         # YOLOv8 model (if applicable)
│-- 📂 Exploratory Code
│   │-- Experiment.ipynb   # Jupyter notebook containing trial code
│   │-- car.jpg            # Test image used for experimental code
```

## 🚀 Project Description
The project identifies and extracts license plates from images using image processing techniques and OCR. It follows these steps:

1. Convert the image to grayscale.
2. Apply noise reduction and edge detection.
3. Find contours and detect a 4-point region (likely the license plate).
4. Extract and crop the license plate area.
5. Use **EasyOCR** to recognize the text from the cropped plate.
6. Overlay the extracted text on the original image.

## 🛠️ Dependencies
To run this project, you need the following Python libraries:

```bash
pip install opencv-python numpy imutils easyocr matplotlib
```

## 📜 Usage
### Run Jupyter Notebook (Recommended)
```bash
jupyter notebook NPD.ipynb
```

### Run Python Script
```bash
python NPD.py
```

Make sure the test images (image1.jpg, image2.jpg, etc.) are present in the **Main Code** folder.

## 🖼️ Sample Workflow
### 1. Input Image
- The input image should contain a vehicle with a visible license plate.

### 2. Edge Detection & Contours
- The script detects edges and finds potential license plate regions.

### 3. Extracted License Plate
- The detected license plate area is extracted and processed.

### 4. OCR Text Detection
- The text is recognized and displayed on the image.

## 📌 Notes
- The **Exploratory Code** folder contains initial trials (`Experiment.ipynb`), which led to the final implementation.
- The **Main Code** folder contains the final working code (`NPD.ipynb` and `NPD.py`).
- The **car.jpg** image was used for experimental purposes only.

## 🎯 Future Enhancements
- Improve OCR accuracy by fine-tuning preprocessing techniques.
- Integrate YOLOv8 (`yolov8n.pt`) for real-time license plate detection.
- Implement support for multiple license plates in a single image.

---

**📧 For questions or contributions, feel free to open an issue or submit a pull request!** 🚀
