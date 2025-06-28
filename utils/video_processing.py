import cv2
import numpy as np
from .green_screen_detection import create_green_screen_mask

def fit_video_to_mask(video_frame, mask):
    """Menyesuaikan video frame dengan bentuk mask green screen."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return video_frame, (0, 0, video_frame.shape[1], video_frame.shape[0]), None
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    resized_video = cv2.resize(video_frame, (w, h))
    
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    adjusted_contour = largest_contour - [x, y]
    cv2.fillPoly(contour_mask, [adjusted_contour], 255)
    
    masked_video = cv2.bitwise_and(resized_video, resized_video, mask=contour_mask)
    
    alpha = contour_mask.astype(np.float32) / 255.0
    alpha = np.stack([alpha] * 3, axis=-1)
    
    background = np.zeros_like(masked_video)
    blended = (masked_video * alpha + background * (1 - alpha)).astype(np.uint8)
    
    return blended, (x, y, w, h), contour_mask

def process_frame_with_green_screen(background_frame, video_frame, template_mask):
    """Memproses frame dengan mengganti green screen dengan video."""
    mask = create_green_screen_mask(background_frame)
    fitted_video, bbox, contour_mask = fit_video_to_mask(video_frame, mask)
    x, y, w, h = bbox
    
    result = background_frame.copy()
    
    if fitted_video is not None and contour_mask is not None:
        mask_3ch = cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        roi = result[y:y+h, x:x+w]
        blended_roi = (fitted_video * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
        result[y:y+h, x:x+w] = blended_roi
    
    return result