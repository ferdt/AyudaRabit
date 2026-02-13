from flask import Flask, request, jsonify
from flask_cors import CORS
from paddleocr import PaddleOCR
import base64
import numpy as np
import cv2
import time

import json
import os
import traceback

# New OCR Engines
try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None
    print("RapidOCR not found. Install with: pip install rapidocr_onnxruntime")

try:
    import pytesseract
    from PIL import Image
    import io
except ImportError:
    pytesseract = None
    print("Pytesseract not found. Install with: pip install pytesseract Pillow")


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load Configuration
CONFIG_FILE = 'config.json'
DEFAULT_CONFIG = {
    "preprocessing": {
        "apply_contrast": False,
        "contrast_alpha": 1.0,
        "contrast_beta": 0,
        "apply_gray": False,
        "apply_threshold": False,
        "threshold_block_size": 15,
        "threshold_c": 5
    },
    "paddleocr": {
        "lang": "es",
        "use_angle_cls": True,
        "enable_mkldnn": False
    }
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG

config = load_config()
ocr_config = config.get('paddleocr', {})

print(f"Loading PaddleOCR with config: {ocr_config}")

# Initialize OCR Engines
paddle_ocr_engine = PaddleOCR(**ocr_config)

rapid_ocr_engine = None
if RapidOCR:
    rapid_ocr_engine = RapidOCR()
    print("RapidOCR initialized.")

# Check Tesseract
tesseract_available = False
if pytesseract:
    try:
        # Check if tesseract is in path or configured
        # You might need to set pytesseract.pytesseract.tesseract_cmd if it's not in PATH
        # e.g. pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        tesseract_version = pytesseract.get_tesseract_version()
        print(f"Tesseract found: version {tesseract_version}")
        tesseract_available = True
    except Exception as e:
        print(f"Tesseract not found or not configured: {e}")


# --- Helper Functions for Image Rectification ---
# ... (Keep existing helper functions: order_points, four_point_transform, detect_and_crop_to_content, apply_sharpening, sort_boxes) ...

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # return the warped image
    return warped

def detect_and_crop_to_content(image):
    # Returns (success, processed_image)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        # Find contours
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        if not cnts:
            return False, image

        # Filter contours by area to ignore noise
        min_area = 1000 # Adjust based on image resolution
        large_cnts = [c for c in cnts if cv2.contourArea(c) > min_area]

        if not large_cnts:
             print("No large contours found. Returning original.")
             return False, image

        # Find the bounding rectangle that encloses ALL large contours
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf

        for c in large_cnts:
            x, y, w, h = cv2.boundingRect(c)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # Add a small padding
        padding = 10
        h_img, w_img = image.shape[:2]
        
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w_img, x_max + padding)
        y_max = min(h_img, y_max + padding)

        # Crop
        cropped = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        
        print(f"Content detected. Cropped to: {x_min},{y_min} - {x_max},{y_max}")
        return True, cropped

    except Exception as e:
        print(f"Error in content cropping: {e}")
        return False, image

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def sort_boxes(dt_boxes, matched_texts, matched_scores):
    """
    Sort boxes by Y-coordinate first (with a tolerance/threshold) to group into rows,
    then by X-coordinate within each row.
    """
    if dt_boxes is None or len(dt_boxes) == 0:
        return [], [], []

    items = []
    for i in range(len(dt_boxes)):
        box = dt_boxes[i]
        
        # Normalize box shape
        box = np.array(box, dtype=np.float32)
        if box.ndim == 1:
            if box.size == 8:
                 box = box.reshape((4, 2))
            elif box.size == 4:
                # Assume [xmin, ymin, xmax, ymax]
                x_min, y_min, x_max, y_max = box
                box = np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ], dtype=np.float32)
            else:
                 # Unexpected shape, try to infer or skip
                 print(f"Skipping box with unexpected shape: {box.shape}")
                 continue
        
        items.append({
            'box': box,
            'text': matched_texts[i],
            'score': matched_scores[i] if matched_scores else 1.0,
            'y_top': min(box[0][1], box[1][1]), # Top Y
            'x_left': min(box[0][0], box[3][0]) # Left X
        })

    # Sort primarily by Y
    items.sort(key=lambda x: x['y_top'])

    # Group into lines
    lines = []
    current_line = []
    
    if items:
        current_line.append(items[0])
        for i in range(1, len(items)):
            item = items[i]
            prev_item = current_line[-1]
            
            prev_height = abs(prev_item['box'][2][1] - prev_item['box'][0][1])
            threshold = prev_height * 0.5 
            
            if abs(item['y_top'] - prev_item['y_top']) < threshold:
                current_line.append(item)
            else:
                lines.append(current_line)
                current_line = [item]
        lines.append(current_line)

    # Sort each line by X and flatten
    sorted_items = []
    for line in lines:
        line.sort(key=lambda x: x['x_left'])
        sorted_items.extend(line)

    sorted_boxes = [x['box'] for x in sorted_items]
    sorted_texts = [x['text'] for x in sorted_items]
    sorted_scores = [x['score'] for x in sorted_items]

    return sorted_boxes, sorted_texts, sorted_scores


@app.route('/ocr', methods=['POST'])
def process_image():
    global config
    config = load_config()
    default_prep_config = config.get('preprocessing', {})

    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Merge request options with default config
        request_prep = data.get('preprocessing', {})
        prep_config = {**default_prep_config, **request_prep}

        image_data = data['image']
        ocr_engine_name = data.get('ocr_engine', 'rapid') # Default to rapid
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
             return jsonify({'error': 'Failed to decode image'}), 400

        print(f"Applying preprocessing: {prep_config}")
        img_to_process = img

        # 1. Deskew
        # 1. Content Detection (Crop to relevant area)
        if prep_config.get('apply_deskew', False):
             # We use the 'apply_deskew' flag to trigger this new safer crop logic
             success, cropped_img = detect_and_crop_to_content(img_to_process)
             if success:
                 img_to_process = cropped_img

        # 1.5 Resize (Optional)
        if prep_config.get('apply_resize', False):
            target_h = prep_config.get('resize_height', 2000)
            h, w = img_to_process.shape[:2]
            
            if h > target_h:
                scale = target_h / h
                new_w = int(w * scale)
                img_to_process = cv2.resize(img_to_process, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
                print(f"Downscaled image to height {target_h} (scale {scale:.2f}) -> {new_w}x{target_h}")
            else:
                 print(f"Image height {h} <= target {target_h}. Skipping resize.")

        # 2. Contrast
        if prep_config.get('apply_contrast', False):
            alpha = prep_config.get('contrast_alpha', 1.5) 
            beta = prep_config.get('contrast_beta', 0)     
            img_to_process = cv2.convertScaleAbs(img_to_process, alpha=alpha, beta=beta)

        # 3. Sharpening
        if prep_config.get('apply_sharpening', False):
            img_to_process = apply_sharpening(img_to_process)

        # 4. Grayscale
        if prep_config.get('apply_gray', False):
             if len(img_to_process.shape) == 3:
                img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
                img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)

        # 5. Thresholding
        if prep_config.get('apply_threshold', False):
            gray_for_thresh = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
            block_size = int(prep_config.get('threshold_block_size', 15))
            c_val = int(prep_config.get('threshold_c', 5))
            
            # Block size must be odd
            if block_size % 2 == 0:
                block_size += 1
                
            img_to_process = cv2.adaptiveThreshold(
                gray_for_thresh, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                block_size, 
                c_val
            )
            img_to_process = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)

        # --- OCR ---
        print(f"Running OCR Engine: {ocr_engine_name}")
        detected_text = ""
        debug_img = img_to_process.copy()
        
        start_time = time.time()

        if ocr_engine_name == 'paddle':
            # Use predict() for consistency
            result = paddle_ocr_engine.predict(img_to_process)
            
            if result:
                first_item = result[0]
                
                def get_val(obj, key):
                    if isinstance(obj, dict):
                        return obj.get(key)
                    elif hasattr(obj, key):
                        return getattr(obj, key)
                    return None

                rec_texts = get_val(first_item, 'rec_texts')
                rec_boxes = get_val(first_item, 'rec_boxes')
                rec_scores = get_val(first_item, 'rec_scores')
                
                if rec_texts is not None and rec_boxes is not None:
                    # Sort Results
                    rec_boxes, rec_texts, rec_scores = sort_boxes(rec_boxes, rec_texts, rec_scores)
                    
                    for i in range(len(rec_texts)):
                        text = rec_texts[i]
                        box = rec_boxes[i]
                        detected_text += text + "\n"
                        
                        points = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(debug_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

                elif isinstance(first_item, list):
                     print("Legacy format detected (not sorted via new logic)")
                     # Simple fallback
                     for line in first_item:
                        try:
                            text = line[1][0]
                            detected_text += text + "\n"
                            coords = line[0]
                            points = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(debug_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                        except:
                            pass
            else:
                print("No text detected.")

        elif ocr_engine_name == 'rapid':
            if not rapid_ocr_engine:
                 return jsonify({'error': 'RapidOCR is not available. Please install dependencies.'}), 500
            
            # RapidOCR expects image path, bytes, or numpy array
            result, elapse = rapid_ocr_engine(img_to_process)
            
            if result:
                # result format: [[box, text, score], ...]
                boxes = [line[0] for line in result]
                texts = [line[1] for line in result]
                scores = [line[2] for line in result]
                
                # Use shared sort logic
                sorted_boxes, sorted_texts, _ = sort_boxes(boxes, texts, scores)

                for i in range(len(sorted_texts)):
                    text = sorted_texts[i]
                    box = sorted_boxes[i]
                    detected_text += text + "\n"
                    
                    # Box format in RapidOCR is usually [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    points = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(debug_img, [points], isClosed=True, color=(0, 0, 255), thickness=2)
            else:
                print("No text detected by RapidOCR.")
        
        elif ocr_engine_name == 'tesseract':
            if not tesseract_available:
                return jsonify({'error': 'Tesseract is not available or configured.'}), 500
            
            # Tesseract needs PIL Image usually, or numpy array
            # Use raw string output for simplicity first, or data dict for boxes
            # config='--psm 6' (Assume single uniform block of text) or 3 (Fully automatic)
            # Spanish language
            custom_config = r'--oem 3 --psm 6' 
            detected_text = pytesseract.image_to_string(img_to_process, lang='spa', config=custom_config)
            
            # For visualization, we can use image_to_data
            d = pytesseract.image_to_data(img_to_process, output_type=pytesseract.Output.DICT, lang='spa')
            n_boxes = len(d['text'])
            for i in range(n_boxes):
                if d['conf'][i] > 0 and d['text'][i].strip():
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 2)


        else:
             return jsonify({'error': f'Unknown OCR engine: {ocr_engine_name}'}), 400

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"OCR Execution Time: {execution_time:.4f} seconds")

        _, buffer = cv2.imencode('.jpg', debug_img)
        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'text': detected_text,
            'processed_image': f"data:image/jpeg;base64,{processed_image_b64}",
            'execution_time': execution_time
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting OCR Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
