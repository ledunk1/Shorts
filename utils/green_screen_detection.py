import cv2
import numpy as np

def create_green_screen_mask(image):
    """Membuat mask untuk area green screen dengan deteksi yang lebih akurat."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_green1 = np.array([35, 40, 40])
    upper_green1 = np.array([85, 255, 255])
    
    lower_green2 = np.array([25, 30, 30])
    upper_green2 = np.array([95, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask

def find_greenscreen_area(image_path):
    """Mendeteksi area hijau pada gambar dan mengembalikan kotak pembatasnya."""
    try:
        image = cv2.imread(image_path)
        if image is None: 
            return None
        mask = create_green_screen_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: 
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest_contour)
    except Exception as e:
        print(f"Error saat mendeteksi green screen: {e}")
        return None