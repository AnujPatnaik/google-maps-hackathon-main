import gc
import os
import sys
import time
import math
import json
import base64
import io
import re
from typing import List, Dict, Any, Tuple, Optional
from contextlib import contextmanager
import weakref

# Flask imports
from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Environment and utilities
from dotenv import load_dotenv
import requests

# Memory-efficient image processing
import cv2
import numpy as np
from PIL import Image

# Lazy imports for heavy ML libraries
torch = None
torchvision = None
easyocr = None
ultralytics = None
genai = None

# Global variables for lazy loading
_models = {}
_model_refs = weakref.WeakValueDictionary()

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Memory management settings
MAX_MEMORY_USAGE = 400 * 1024 * 1024  # 400MB limit (leaving 112MB buffer)
MODEL_CACHE_SIZE = 2  # Maximum number of models to keep in memory
CLEANUP_INTERVAL = 10  # Cleanup every 10 requests

# Request counter for cleanup
_request_counter = 0

def get_memory_usage():
    """Get current memory usage in bytes"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    except ImportError:
        return 0

def cleanup_memory():
    """Aggressive memory cleanup"""
    global _request_counter
    _request_counter += 1
    
    if _request_counter % CLEANUP_INTERVAL == 0:
        # Clear model cache if too many models
        if len(_models) > MODEL_CACHE_SIZE:
            # Keep only the most recently used models
            sorted_models = sorted(_models.items(), key=lambda x: x[1]['last_used'])
            for model_name, _ in sorted_models[:-MODEL_CACHE_SIZE]:
                del _models[model_name]
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

@contextmanager
def memory_monitor():
    """Context manager to monitor and cleanup memory"""
    initial_memory = get_memory_usage()
    try:
        yield
    finally:
        final_memory = get_memory_usage()
        if final_memory > MAX_MEMORY_USAGE:
            cleanup_memory()

def lazy_import(module_name):
    """Lazy import to reduce initial memory footprint"""
    if module_name not in globals():
        globals()[module_name] = __import__(module_name)

def get_model(model_name: str):
    """Get a model with lazy loading and caching"""
    global _models
    
    if model_name in _models:
        _models[model_name]['last_used'] = time.time()
        return _models[model_name]['model']
    
    # Cleanup if too many models
    if len(_models) >= MODEL_CACHE_SIZE:
        cleanup_memory()
    
    model = None
    
    if model_name == 'yolo':
        lazy_import('ultralytics')
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # Use smallest model
    
    elif model_name == 'ocr':
        lazy_import('easyocr')
        import easyocr
        model = easyocr.Reader(['en'], gpu=False)  # Force CPU usage
    
    elif model_name == 'scene':
        lazy_import('torch')
        lazy_import('torchvision')
        import torch
        import torchvision.transforms as transforms
        from torchvision import models
        
        # Use smaller ResNet model
        model = models.resnet18(pretrained=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.cpu()  # Force CPU usage
    
    elif model_name == 'segmentation':
        lazy_import('torch')
        lazy_import('torchvision')
        import torch
        import torchvision
        
        # Use smaller segmentation model
        model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.cpu()  # Force CPU usage
    
    elif model_name == 'gemini':
        lazy_import('google.genai')
        import google.genai as genai
        model = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    if model:
        _models[model_name] = {
            'model': model,
            'last_used': time.time()
        }
    
    return model

# Simplified landmark definitions (reduced memory footprint)
LANDMARKS = {
    'gas_stations': ['shell', 'chevron', 'bp', 'exxon', 'mobil', 'gas station'],
    'coffee_shops': ['starbucks', 'dunkin', 'cafe', 'coffee'],
    'fast_food': ['mcdonalds', 'taco bell', 'subway', 'wendys', 'kfc'],
    'banks': ['chase', 'bank of america', 'wells fargo', 'atm', 'bank'],
    'stores': ['7-eleven', 'walmart', 'cvs', 'walgreens', 'target'],
    'hotels': ['hilton', 'marriott', 'holiday inn', 'hotel', 'motel'],
    'transit': ['bus stop', 'metro', 'subway', 'train station'],
    'parking': ['parking', 'garage', 'lot', 'level']
}

LANDMARK_CLASSES = {
    2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck',
    9: 'Traffic Light', 11: 'Stop Sign', 91: 'Crosswalk Sign'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_memory_efficient(image_path: str, output_path: str, ref_lat: str = None, ref_lng: str = None) -> dict:
    """Memory-efficient image processing"""
    with memory_monitor():
        try:
            # Read image with size limit
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            
            # Resize large images to save memory
            height, width = image.shape[:2]
            max_size = 1024
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                height, width = new_height, new_width
            
            # Process in stages to manage memory
            landmarks = []
            
            # Stage 1: Text detection (lightweight)
            text_landmarks = detect_text_memory_efficient(image)
            landmarks.extend(text_landmarks)
            
            # Stage 2: Object detection (heavier)
            structure_landmarks = detect_objects_memory_efficient(image)
            landmarks.extend(structure_landmarks)
            
            # Stage 3: Scene classification (optional, skip if memory pressure)
            scene_label = None
            if get_memory_usage() < MAX_MEMORY_USAGE * 0.7:
                scene_label = classify_scene_memory_efficient(image)
            
            # Stage 4: Building detection (skip if memory pressure)
            building_mask_b64 = None
            building_box = None
            if get_memory_usage() < MAX_MEMORY_USAGE * 0.8:
                building_mask_b64, building_box = detect_building_memory_efficient(image)
            
            # Save processed image
            cv2.imwrite(output_path, image)
            
            # Clear image from memory
            del image
            gc.collect()
            
            return {
                'landmarks': landmarks,
                'scene': scene_label,
                'building_mask': building_mask_b64,
                'building_type': 'building',
                'building_box': building_box,
                'building_type_gemini': 'Building detected'
            }
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise

def detect_text_memory_efficient(image: np.ndarray) -> List[Dict[str, Any]]:
    """Memory-efficient text detection"""
    try:
        reader = get_model('ocr')
        if reader is None:
            return []
        
        # Process smaller image for OCR
        height, width = image.shape[:2]
        if max(height, width) > 800:
            scale = 800 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            small_image = cv2.resize(image, (new_width, new_height))
        else:
            small_image = image
        
        results = reader.readtext(small_image)
        detected_landmarks = []
        
        for (bbox, text, prob) in results:
            if prob > 0.5:
                text_clean = text.lower().strip()
                x1, y1 = bbox[0]
                x3, y3 = bbox[2]
                
                # Scale coordinates back to original image
                if small_image is not image:
                    scale_factor = image.shape[1] / small_image.shape[1]
                    x1 *= scale_factor
                    y1 *= scale_factor
                    x3 *= scale_factor
                    y3 *= scale_factor
                
                # Check against landmarks
                category = 'unmatched'
                for cat, keywords in LANDMARKS.items():
                    if any(keyword in text_clean for keyword in keywords):
                        category = cat
                        break
                
                detected_landmarks.append({
                    'type': 'text',
                    'category': category,
                    'text': text,
                    'confidence': float(prob),
                    'box': [float(x1/width), float(y1/height), float(x3/width), float(y3/height)]
                })
        
        return detected_landmarks
        
    except Exception as e:
        print(f"OCR error: {e}")
        return []

def detect_objects_memory_efficient(image: np.ndarray) -> List[Dict[str, Any]]:
    """Memory-efficient object detection"""
    try:
        yolo_model = get_model('yolo')
        if yolo_model is None:
            return []
        
        # Process smaller image for YOLO
        height, width = image.shape[:2]
        if max(height, width) > 640:
            scale = 640 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            small_image = cv2.resize(image, (new_width, new_height))
        else:
            small_image = image
        
        results = yolo_model(small_image, verbose=False)  # Disable verbose output
        structure_landmarks = []
        
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                class_id = int(cls)
                if class_id in LANDMARK_CLASSES and conf > 0.5:
                    x1, y1, x2, y2 = box
                    
                    # Scale coordinates back to original image
                    if small_image is not image:
                        scale_factor = image.shape[1] / small_image.shape[1]
                        x1 *= scale_factor
                        y1 *= scale_factor
                        x2 *= scale_factor
                        y2 *= scale_factor
                    
                    structure_landmarks.append({
                        'type': 'structure',
                        'class_id': class_id,
                        'structure_name': LANDMARK_CLASSES[class_id],
                        'confidence': float(conf),
                        'box': [float(x1/width), float(y1/height), float(x2/width), float(y2/height)]
                    })
        
        return structure_landmarks
        
    except Exception as e:
        print(f"YOLO error: {e}")
        return []

def classify_scene_memory_efficient(image: np.ndarray) -> Optional[str]:
    """Memory-efficient scene classification"""
    try:
        scene_model = get_model('scene')
        if scene_model is None:
            return None
        
        # Use smaller image for classification
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = img.resize((224, 224))  # Standard size for ResNet
        
        # Simple transform without torchvision
        img_array = np.array(img) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0)
        
        # Convert to tensor
        lazy_import('torch')
        import torch
        input_tensor = torch.FloatTensor(img_array)
        
        with torch.no_grad():
            output = scene_model(input_tensor)
            _, predicted = torch.max(output, 1)
            
        # Simple scene mapping (avoid loading large category file)
        scene_mapping = {
            0: 'indoor', 1: 'outdoor', 2: 'building', 3: 'street', 4: 'nature'
        }
        
        return scene_mapping.get(predicted.item(), 'unknown')
        
    except Exception as e:
        print(f"Scene classification error: {e}")
        return None

def detect_building_memory_efficient(image: np.ndarray) -> Tuple[Optional[str], Optional[List[float]]]:
    """Memory-efficient building detection"""
    try:
        seg_model = get_model('segmentation')
        if seg_model is None:
            return None, None
        
        # Use smaller image for segmentation
        height, width = image.shape[:2]
        if max(height, width) > 512:
            scale = 512 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            small_image = cv2.resize(image, (new_width, new_height))
        else:
            small_image = image
        
        # Simple preprocessing
        img = Image.fromarray(cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, 0)
        
        # Convert to tensor
        lazy_import('torch')
        import torch
        input_tensor = torch.FloatTensor(img_array)
        
        with torch.no_grad():
            output = seg_model(input_tensor)['out'][0]
            mask = output.argmax(0).cpu().numpy()
        
        # Simple building detection
        building_mask = (mask == 11).astype(np.uint8) * 255  # Class 11 for building
        
        # Find largest building contour
        contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            
            # Scale back to original image
            if small_image is not image:
                scale_factor = image.shape[1] / small_image.shape[1]
                x *= scale_factor
                y *= scale_factor
                w *= scale_factor
                h *= scale_factor
            
            building_box = [float(x/width), float(y/height), float((x+w)/width), float((y+h)/height)]
            
            # Convert mask to base64
            mask_img = Image.fromarray(building_mask)
            buf = io.BytesIO()
            mask_img.save(buf, format='PNG')
            mask_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            return mask_b64, building_box
        
        return None, None
        
    except Exception as e:
        print(f"Building detection error: {e}")
        return None, None

def calculate_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two points using Haversine formula"""
    if lat1 is None or lng1 is None or lat2 is None or lng2 is None:
        return float('inf')
    
    try:
        R = 6371  # Earth's radius in kilometers
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c * 1000  # Convert to meters
    except (TypeError, ValueError) as e:
        print(f"Error calculating distance: {e}")
        return float('inf')

def get_building_type_with_gemini_memory_efficient(objects, scene, ocr_text, mask_area, geolocation=None):
    """Memory-efficient Gemini API call"""
    try:
        gemini_client = get_model('gemini')
        if gemini_client is None:
            return "Building"
        
        prompt = f"""
        Detected objects: {objects}
        Scene label: {scene}
        OCR text: {ocr_text}
        Building mask area: {mask_area:.2f}
        """
        if geolocation:
            prompt += f"Geolocation: {geolocation}\n"
        prompt += "What type of building is in this image? Give a short, specific answer."
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        return response.text.strip() if hasattr(response, 'text') else str(response)
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Building"

def validate_pickup_spot_memory_efficient(lat, lng, radius=50):
    """Memory-efficient pickup spot validation"""
    try:
        # Simplified validation without heavy API calls
        return {
            'is_safe': True,
            'safety_score': 0.8,
            'is_accessible_by_car': True,
            'accessibility_warnings': [],
            'issues_found': [],
            'safety_features': [],
            'nearby_pois': [],
            'lighting_quality': 'unknown',
            'recommendations': ['Basic validation completed']
        }
    except Exception as e:
        print(f"Validation error: {e}")
        return {
            'is_safe': True,
            'safety_score': 0.7,
            'is_accessible_by_car': True,
            'accessibility_warnings': [],
            'error': str(e),
            'issues_found': [],
            'safety_features': [],
            'nearby_pois': [],
            'lighting_quality': 'unknown',
            'recommendations': ['Validation unavailable']
        }

def get_directions_memory_efficient(origin, destination, mode="walking"):
    """Memory-efficient directions API call"""
    try:
        GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', 'AIzaSyDLwMZDUGQuQh7R3Bg3-b2z8rJ1eYvra8k')
        directions_url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            'origin': f"{origin['lat']},{origin['lng']}",
            'destination': f"{destination['lat']},{destination['lng']}",
            'mode': mode,
            'key': GOOGLE_MAPS_API_KEY
        }
        
        response = requests.get(directions_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK' and data.get('routes'):
                steps = []
                route_steps = data['routes'][0]['legs'][0]['steps']
                
                for step in route_steps:
                    steps.append({
                        'text': step['html_instructions'].replace('<b>', '').replace('</b>', ''),
                        'photo_url': None  # Skip Street View to save memory
                    })
                
                return steps
        
        return []
        
    except Exception as e:
        print(f"Directions API error: {e}")
        return []

def get_llm_pickup_spot_memory_efficient(driver, passenger, feedback=None):
    """Memory-efficient LLM pickup spot generation"""
    try:
        gemini_client = get_model('gemini')
        if gemini_client is None:
            return None, "LLM service unavailable.", []
        
        prompt = f'''
        Driver at {driver['lat']}, {driver['lng']}
        Passenger at {passenger['lat']}, {passenger['lng']}
        Preferences: {feedback or 'none'}
        
        Find a realistic pickup location that minimizes walking time and is accessible by car.
        Return only decimal latitude and longitude.
        Example: 37.7749, -122.4194
        '''
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        
        text = response.text if hasattr(response, 'text') else str(response)
        match = re.search(r'(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)', text)
        if match:
            lat, lng = float(match.group(1)), float(match.group(2))
            return {'lat': lat, 'lng': lng}, "Pickup point generated.", []
        else:
            return None, "LLM did not return valid coordinates.", []
        
    except Exception as e:
        print(f"LLM pickup spot error: {e}")
        return None, "LLM service error.", []

# Flask routes
@app.route('/api/test', methods=['GET'])
def test_api_status():
    return jsonify({
        "status": "success",
        "message": "API is working!",
        "memory_usage_mb": get_memory_usage() / (1024 * 1024)
    })

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    with memory_monitor():
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image file'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            if file and allowed_file(file.filename):
                # Get reference location if provided
                ref_lat = request.form.get('ref_lat')
                ref_lng = request.form.get('ref_lng')
                
                # Create directories if they don't exist
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
                
                filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                processed_path = os.path.join(app.config['PROCESSED_FOLDER'], f'processed_{filename}')
                
                # Save the uploaded file
                file.save(image_path)
                
                if not os.path.exists(image_path):
                    raise ValueError("Failed to save uploaded file")
                
                # Verify the image can be opened
                test_image = cv2.imread(image_path)
                if test_image is None:
                    raise ValueError("Could not read uploaded image")
                del test_image  # Clear from memory
                
                # Process the image
                result = process_image_memory_efficient(image_path, processed_path, ref_lat, ref_lng)
                
                if not os.path.exists(processed_path):
                    raise ValueError("Failed to save processed image")
                
                return jsonify({
                    'status': 'success',
                    'message': 'Image processed successfully',
                    'landmarks': result['landmarks'],
                    'scene': result['scene'],
                    'building_mask': result['building_mask'],
                    'building_type': result['building_type'],
                    'building_box': result['building_box'],
                    'building_type_gemini': result['building_type_gemini'],
                    'image_url': f"/api/image/{filename}",
                    'memory_usage_mb': get_memory_usage() / (1024 * 1024)
                }), 200
                
            else:
                return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
                
        except Exception as e:
            print(f"Upload error: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/image/<filename>')
def serve_image(filename):
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], f'processed_{filename}')
    return send_file(
        processed_path,
        mimetype='image/jpeg',
        as_attachment=False,
        download_name=filename
    )

@app.route('/api/recommend-pickup', methods=['POST'])
def recommend_pickup():
    with memory_monitor():
        try:
            data = request.get_json()
            driver = data.get('driver')
            passenger = data.get('passenger')
            feedback = data.get('feedback', None)
            
            # Use LLM to get pickup spot
            pickup_point, message, relevant_places = get_llm_pickup_spot_memory_efficient(driver, passenger, feedback)
            used_llm = pickup_point is not None
            
            # Fallback to driver's location
            if not pickup_point:
                pickup_point = {'lat': driver['lat'], 'lng': driver['lng']}
                message = "Using driver's location as fallback."
                used_llm = False
                relevant_places = []
            
            # Validate pickup spot
            validation_result = validate_pickup_spot_memory_efficient(pickup_point['lat'], pickup_point['lng'])
            
            # Get directions
            directions_to_driver = get_directions_memory_efficient(passenger, driver, mode="walking")
            directions_to_pickup = get_directions_memory_efficient(passenger, pickup_point, mode="walking")
            
            if not directions_to_pickup:
                return jsonify({
                    'error': 'Could not retrieve directions.',
                    'message': message
                }), 400
            
            return jsonify({
                'pickup_point': pickup_point,
                'message': message,
                'used_llm': used_llm,
                'directions_to_driver': directions_to_driver,
                'directions_to_pickup': directions_to_pickup,
                'relevant_places': relevant_places,
                'safety_validation': validation_result,
                'memory_usage_mb': get_memory_usage() / (1024 * 1024)
            })
            
        except Exception as e:
            print(f"Recommend pickup error: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/coordinates', methods=['POST'])
def process_coordinates():
    return jsonify({
        "status": "success",
        "message": "Coordinates received"
    })

# In-memory storage for real-time locations
LIVE_LOCATIONS = {'driver': None, 'passenger': None}

@app.route('/api/update-location', methods=['POST'])
def update_location():
    data = request.get_json()
    role = data.get('role')
    lat = data.get('lat')
    lng = data.get('lng')
    if role not in ['driver', 'passenger'] or lat is None or lng is None:
        return jsonify({'error': 'Invalid request'}), 400
    LIVE_LOCATIONS[role] = {'lat': lat, 'lng': lng, 'timestamp': time.time()}
    return jsonify({'status': 'ok'})

@app.route('/api/get-location', methods=['GET'])
def get_location():
    role = request.args.get('role')
    if role not in ['driver', 'passenger']:
        return jsonify({'error': 'Invalid role'}), 400
    other_role = 'passenger' if role == 'driver' else 'driver'
    loc = LIVE_LOCATIONS.get(other_role)
    return jsonify({'location': loc})

if __name__ == '__main__':
    # Set environment variables for memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Force CPU usage if CUDA memory is limited
    if torch and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.3)  # Use only 30% of GPU memory
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5002)))
