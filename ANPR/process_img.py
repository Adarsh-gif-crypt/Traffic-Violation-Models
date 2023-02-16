import cv2
import numpy as np
import pytesseract

class img_process:
    def __init__(self):
        
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def process_image(self, cropped_image):
        
        self.color_dst = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        self.pixel_vals = self.color_dst.reshape((-1,3))
        self.reshaped_pixel = np.float32(self.pixel_vals)
        
        self.criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        self.k = 2
        
        self.retval, self.labels, self.centers = cv2.kmeans(self.reshaped_pixel, self.k, None, self.criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        self.centers = np.uint8(self.centers)
        self.meta = self.centers[self.labels.flatten()]

        self.src = self.meta.reshape(self.color_dst.shape)

        self.kernel = np.ones((5,5),np.float32)/25
        self.dst = cv2.filter2D(self.color_dst, -1, self.kernel)

        return self.dst
    
    def note(self):
        print('Here')
    
    def extract_text(self, processed_img):
        
        self.text_extract = pytesseract.image_to_string(processed_img)
        return self.text_extract