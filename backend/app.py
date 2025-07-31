from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS
from dotenv import load_dotenv
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import json

import easyocr
import re
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

import torchvision
import base64
import io
import matplotlib.pyplot as plt

import google.genai as genai
from google.genai import types

import requests
import time
from dotenv import load_dotenv
import os

load_dotenv()
import math


gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_building_type_with_gemini(objects, scene, ocr_text, mask_area, geolocation=None):
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

# Initialize OCR reader
reader = easyocr.Reader(['en'])

yolov8_model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' for better accuracy if available

# Define common landmarks and their variations
LANDMARKS = {
    'gas_stations': [
        'shell', 'chevron', 'bp', 'exxon', 'mobil', 'texaco', 'gas station', 'fuel'
    ],
    'coffee_shops': [
        'starbucks', 'dunkin', 'tim hortons', 'cafe', 'coffee'
    ],
    'fast_food': [
        'mcdonalds', 'taco bell', 'subway', 'wendys', 'kfc', 'burger king',
        'pizza hut', 'dominos', 'chipotle'
    ],
    'banks': [
        'chase', 'bank of america', 'wells fargo', 'citibank', 'atm', 'bank'
    ],
    'stores': [
        '7-eleven', 'walmart', 'whole foods', 'cvs', 'walgreens', 'target',
        'grocery', 'market', 'store'
    ],
    'hotels': [
        'hilton', 'marriott', 'holiday inn', 'hotel', 'motel', 'inn'
    ],
    'transit': [
        'bus stop', 'metro', 'subway', 'train station', 'transit', 'station'
    ],
    'shopping': [
        'mall', 'shopping center', 'plaza', 'outlet', 'food court'
    ],
    'parking': [
        'parking', 'garage', 'lot', 'p1', 'p2', 'p3', 'level'
    ],
    'education': [
        'school', 'university', 'college', 'campus', 'library', 'student'
    ],
    'healthcare': [
        'hospital', 'emergency', 'clinic', 'medical center', 'urgent care'
    ],
    'entertainment': [
        'theater', 'cinema', 'movies', 'concert', 'stadium', 'arena'
    ]
}

# Define landmark classes for structure detection
LANDMARK_CLASSES = {
    # Vehicles (from COCO)
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck',
    
    # Traffic Control (from COCO)
    9: 'Traffic Light',
    11: 'Stop Sign',

    # Custom Detections
    91: 'Crosswalk Sign' # from detect_crosswalk_signs
}

# Add a mapping for notable structures (COCO class IDs to real-world names)
NOTABLE_STRUCTURES = {
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'traffic light',
    11: 'stop sign',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush',
    85: 'skyscraper', # heuristic: very tall building
    86: 'dome', # heuristic: round, large building
    87: 'crane',
    88: 'construction site', # heuristic: crane + unfinished building
    89: 'government building', # heuristic: large, rectangular building
    90: 'apartment building', # heuristic: tall, rectangular building
    91: 'stadium', # heuristic: large, round/oval structure
    92: 'bridge',
    93: 'tower',
    94: 'monument',
    95: 'fountain',
    96: 'plaza',
    97: 'mall',
    98: 'hotel',
    99: 'hospital',
    100: 'library',
    101: 'church',
    102: 'mosque',
    103: 'temple',
    104: 'synagogue',
    105: 'school',
    106: 'university',
    107: 'museum',
    108: 'theater',
    109: 'cinema',
    110: 'parking garage',
    111: 'gas station',
    112: 'restaurant',
    113: 'cafe',
    114: 'bar',
    115: 'bakery',
    116: 'pharmacy',
    117: 'bank',
    118: 'atm',
    119: 'post office',
    120: 'police station',
    121: 'fire station',
    122: 'embassy',
    123: 'courthouse',
    124: 'city hall',
    125: 'market',
    126: 'supermarket',
    127: 'grocery store',
    128: 'convenience store',
    129: 'hardware store',
    130: 'bookstore',
    131: 'clothing store',
    132: 'shoe store',
    133: 'jewelry store',
    134: 'toy store',
    135: 'furniture store',
    136: 'electronics store',
    137: 'sports store',
    138: 'music store',
    139: 'pet store',
    140: 'florist',
    141: 'laundry',
    142: 'dry cleaner',
    143: 'hair salon',
    144: 'barber shop',
    145: 'spa',
    146: 'gym',
    147: 'swimming pool',
    148: 'playground',
    149: 'park',
    150: 'zoo',
    151: 'aquarium',
    152: 'amusement park',
    153: 'theme park',
    154: 'water park',
    155: 'campground',
    156: 'rv park',
    157: 'marina',
    158: 'pier',
    159: 'beach',
    160: 'mountain',
    161: 'forest',
    162: 'desert',
    163: 'lake',
    164: 'river',
    165: 'waterfall',
    166: 'cave',
    167: 'island',
    168: 'volcano',
    169: 'glacier',
    170: 'cliff',
    171: 'valley',
    172: 'canyon',
    173: 'plain',
    174: 'hill',
    175: 'meadow',
    176: 'field',
    177: 'vineyard',
    178: 'orchard',
    179: 'garden',
    180: 'farm',
    181: 'barn',
    182: 'greenhouse',
    183: 'windmill',
    184: 'lighthouse',
    185: 'castle',
    186: 'palace',
    187: 'fort',
    188: 'ruins',
    189: 'arch',
    190: 'obelisk',
    191: 'pagoda',
    192: 'shrine',
    193: 'torii',
    194: 'minaret',
    195: 'stupa',
    196: 'belltower',
    197: 'wind turbine',
    198: 'solar panel',
    199: 'satellite dish',
    200: 'antenna',
}

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path, output_path):
    # Remove YOLOv4 weights/config download and net loading from process_video and process_image
    # Remove detect_structures function and all its usages
    # Remove any references to yolov4.weights, yolov4.cfg, and coco.names
    # (The actual YOLOv8 integration will be added in the next step)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create video writer with H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    landmarks = []
    frame_count = 0

    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    last_progress = 0

    print(f"Starting video processing: {width}x{height} @ {fps}fps, total frames: {total_frames}")
    landmarks = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frames += 1
        current_progress = int((processed_frames / total_frames) * 100)
        
        # Only update progress when it changes by at least 1%
        if current_progress != last_progress:
            last_progress = current_progress
            print(f"Progress: {current_progress}%")
        
        if processed_frames % 5 != 0:  # Process every 5th frame to improve speed
            continue
            
        try:
            # Detect text-based landmarks
            text_landmarks = detect_text_and_landmarks(frame)
            
            # --- YOLOv8 detection ---
            results = yolov8_model(frame)
            structure_landmarks = []
            for r in results:
                for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    class_id = int(cls)
                    if class_id in LANDMARK_CLASSES:
                        x1, y1, x2, y2 = box
                        structure_landmarks.append({
                            'type': 'structure',
                            'class_id': class_id,
                            'confidence': float(conf),
                            'box': [float(x1/width), float(y1/height), float(x2/width), float(y2/height)]
                        })
            
            # Detect crosswalk signs
            crosswalk_sign_landmarks = detect_crosswalk_signs(frame)

            # Process detected landmarks
            frame_landmarks = []
            
            # Process text landmarks
            for landmark in text_landmarks:
                frame_landmarks.append({
                    'frame': processed_frames,
                    'type': 'text',
                    'text': landmark['text'],
                    'category': landmark['category'],
                    'confidence': landmark['confidence'],
                    'box': landmark['box']
                })
                
                # Draw on frame
                x1 = int(landmark['box'][0] * width)
                y1 = int(landmark['box'][1] * height)
                x2 = int(landmark['box'][2] * width)
                y2 = int(landmark['box'][3] * height)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, landmark['text'], (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Process structure landmarks
            for landmark in structure_landmarks:
                landmark_data = process_landmark(landmark, frame)
                structure_name = landmark_data.get('structure_name', LANDMARK_CLASSES.get(landmark_data['class_id'], 'Unknown Structure'))

                # --- OCR on sign regions for signs ---
                sign_text = None
                if landmark_data['class_id'] in [9, 11, 91]:  # Traffic Light, Stop Sign, Crosswalk Sign
                    x1 = int(landmark_data['box'][0] * width)
                    y1 = int(landmark_data['box'][1] * height)
                    x2 = int(landmark_data['box'][2] * width)
                    y2 = int(landmark_data['box'][3] * height)
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        ocr_results = reader.readtext(roi)
                        if ocr_results:
                            sign_text = ' '.join([text for _, text, prob in ocr_results if prob > 0.5])

                frame_landmarks.append({
                    'frame': processed_frames,
                    'type': 'structure',
                    'class_id': landmark_data['class_id'],
                    'structure_name': structure_name,
                    'confidence': landmark_data['confidence'],
                    'box': landmark_data['box'],
                    'text': sign_text
                })
                
                # Draw on frame
                x1 = int(landmark_data['box'][0] * width)
                y1 = int(landmark_data['box'][1] * height)
                x2 = int(landmark_data['box'][2] * width)
                y2 = int(landmark_data['box'][3] * height)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, structure_name, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Process crosswalk sign landmarks
            for landmark in crosswalk_sign_landmarks:
                structure_name = landmark['structure_name']
                frame_landmarks.append({
                    'frame': processed_frames,
                    'type': 'structure',
                    'class_id': landmark['class_id'],
                    'structure_name': structure_name,
                    'confidence': landmark['confidence'],
                    'box': landmark['box']
                })
                # Draw on frame
                x1 = int(landmark['box'][0] * width)
                y1 = int(landmark['box'][1] * height)
                x2 = int(landmark['box'][2] * width)
                y2 = int(landmark['box'][3] * height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow for signs
                cv2.putText(frame, structure_name, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Add frame landmarks to overall landmarks list if any were detected
            if frame_landmarks:
                landmarks.extend(frame_landmarks)
            
        except Exception as e:
            print(f"Error processing frame {processed_frames}: {str(e)}")
            continue
        
        # Write the processed frame
        out.write(frame)
    
    # Clean up
    cap.release()
    out.release()
    
    return landmarks



def detect_text_and_landmarks(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Detect text in the frame and match it against known landmarks. Always include all detected text regions."""
    detected_landmarks = []
    results = reader.readtext(frame)
    height, width = frame.shape[:2]

    for (bbox, text, prob) in results:
        if prob > 0.5:
            text_clean = text.lower().strip()
            x1, y1 = bbox[0]
            x3, y3 = bbox[2]
            matched = False
            # Check text against known landmarks
            for category, keywords in LANDMARKS.items():
                for keyword in keywords:
                    if keyword in text_clean or text_clean in keyword:
                        detected_landmarks.append({
                            'type': 'text',
                            'category': category,
                            'text': text,
                            'confidence': float(prob),
                            'box': [
                                float(x1/width),
                                float(y1/height),
                                float(x3/width),
                                float(y3/height)
                            ]
                        })
                        matched = True
                        break
                if matched:
                    break
            # Always include unmatched text regions
            if not matched:
                detected_landmarks.append({
                    'type': 'text',
                    'category': 'unmatched',
                    'text': text,
                    'confidence': float(prob),
                    'box': [
                        float(x1/width),
                        float(y1/height),
                        float(x3/width),
                        float(y3/height)
                    ]
                })
    return detected_landmarks

# Remove YOLOv4 weights/config download and net loading from process_video and process_image
# Remove detect_structures function and all its usages
# Remove any references to yolov4.weights, yolov4.cfg, and coco.names
# (The actual YOLOv8 integration will be added in the next step)

def calculate_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two points using Haversine formula"""
    # Check for None values
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

def analyze_pickup_spot_with_gemini(lat, lng):
    """Analyze pickup spot using Gemini AI and Google Maps APIs"""
    
    try:
        # Get Street View image URL
        street_view_url = f"https://maps.googleapis.com/maps/api/streetview"
        params = {
            'size': '600x400',
            'location': f'{lat},{lng}',
            'heading': 0,  # North
            'key': os.getenv('GOOGLE_MAPS_API_KEY', 'AIzaSyDLwMZDUGQuQh7R3Bg3-b2z8rJ1eYvra8k')
        }
        
        # Create prompt for Gemini to analyze the Street View
        prompt = f"""
        Analyze this Street View image for pickup spot safety: {street_view_url}?{requests.compat.urlencode(params)}
        
        Please assess the following safety factors:
        1. Parking restrictions (no parking signs, private property, etc.)
        2. Construction or road work
        3. Safety hazards (poor lighting, dangerous areas)
        4. Accessibility issues (stairs, narrow passages)
        5. Safety features (streetlights, crosswalks, benches)
        
        Return your analysis in this JSON format:
        {{
            "safety_score": 0.85,
            "is_safe": true,
            "issues_found": ["no parking sign visible"],
            "safety_features": ["streetlight nearby", "crosswalk available"],
            "lighting_quality": "good",
            "recommendations": ["Verify parking legality", "Well-lit area suitable for pickup"]
        }}
        
        Safety score should be 0.0-1.0 where:
        - 0.9-1.0: Excellent (well-lit, safe, accessible)
        - 0.7-0.8: Good (adequate safety, minor concerns)
        - 0.5-0.6: Fair (some concerns, proceed with caution)
        - 0.0-0.4: Poor (significant safety issues)
        """
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        # Parse the response
        try:
            analysis_text = response.text.strip() if hasattr(response, 'text') else str(response)
            # Extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                import json
                analysis = json.loads(json_match.group())
                return analysis
            else:
                # Fallback if JSON parsing fails
                return {
                    'safety_score': 0.7,
                    'is_safe': True,
                    'issues_found': [],
                    'safety_features': [],
                    'lighting_quality': 'unknown',
                    'recommendations': ['AI analysis completed - proceed with normal caution']
                }
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            return {
                'safety_score': 0.7,
                'is_safe': True,
                'issues_found': [],
                'safety_features': [],
                'lighting_quality': 'unknown',
                'recommendations': ['Analysis completed - proceed with normal caution']
            }
            
    except Exception as e:
        print(f"Error analyzing pickup spot: {e}")
        return {
            'safety_score': 0.7,
            'is_safe': True,
            'issues_found': [],
            'safety_features': [],
            'lighting_quality': 'unknown',
            'recommendations': ['Analysis unavailable - proceed with caution']
        }

def get_nearby_safety_pois(lat, lng, radius=100):
    """Get nearby safety-related POIs using Google Places API"""
    
    try:
        # Use Google Places API to find nearby safety POIs
        places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            'location': f'{lat},{lng}',
            'radius': radius,
            'type': 'police|hospital|fire_station',
            'key': os.getenv('GOOGLE_MAPS_API_KEY', 'AIzaSyDLwMZDUGQuQh7R3Bg3-b2z8rJ1eYvra8k')
        }
        
        response = requests.get(places_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK':
                pois = []
                for place in data['results']:
                    try:
                        place_lat = place['geometry']['location']['lat']
                        place_lng = place['geometry']['location']['lng']
                        distance = calculate_distance(lat, lng, place_lat, place_lng)
                        pois.append({
                            'name': place['name'],
                            'type': place['types'][0] if place['types'] else 'unknown',
                            'distance': distance
                        })
                    except (KeyError, TypeError) as e:
                        print(f"Error processing place: {e}")
                        continue
                return pois
        
        return []
        
    except Exception as e:
        print(f"Error getting nearby POIs: {e}")
        return []

def validate_car_accessibility(lat, lng):
    """Check if a pickup spot is accessible by car using Google Maps APIs"""
    
    try:
        # Use Google Maps Directions API to check if a car can reach this location
        directions_url = "https://maps.googleapis.com/maps/api/directions/json"
        
        # Test with a nearby point (100m away) to see if car can reach the pickup spot
        test_origin_lat = lat + 0.001  # 100m north
        test_origin_lng = lng + 0.001  # 100m east
        
        params = {
            'origin': f'{test_origin_lat},{test_origin_lng}',
            'destination': f'{lat},{lng}',
            'mode': 'driving',
            'key': os.getenv('GOOGLE_MAPS_API_KEY', 'AIzaSyDLwMZDUGQuQh7R3Bg3-b2z8rJ1eYvra8k')
        }
        
        response = requests.get(directions_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Check if route exists and is accessible by car
            if data['status'] == 'OK' and data['routes']:
                route = data['routes'][0]
                
                # Check for any warnings that might indicate accessibility issues
                warnings = route.get('warnings', [])
                has_accessibility_warnings = any(
                    'pedestrian' in warning.lower() or 
                    'walking' in warning.lower() or
                    'no driving' in warning.lower() or
                    'restricted' in warning.lower()
                    for warning in warnings
                )
                
                # Check if the route uses only driving roads
                legs = route.get('legs', [])
                if legs:
                    steps = legs[0].get('steps', [])
                    has_non_driving_steps = any(
                        step.get('travel_mode') != 'DRIVING' or
                        'pedestrian' in step.get('html_instructions', '').lower()
                        for step in steps
                    )
                    
                    return {
                        'is_accessible': not has_accessibility_warnings and not has_non_driving_steps,
                        'warnings': warnings,
                        'route_exists': True
                    }
            
            return {
                'is_accessible': False,
                'warnings': ['No driving route found'],
                'route_exists': False
            }
        
        return {
            'is_accessible': False,
            'warnings': ['Could not verify accessibility'],
            'route_exists': False
        }
        
    except Exception as e:
        print(f"Error checking car accessibility: {e}")
        return {
            'is_accessible': True,  # Default to accessible if check fails
            'warnings': [],
            'route_exists': True
        }

def validate_pickup_spot(lat, lng, radius=50):
    """Lightweight pickup spot validation using Gemini AI and Google Maps APIs"""
    
    try:
        # Stage 1: Check car accessibility
        accessibility_check = validate_car_accessibility(lat, lng)
        
        # Stage 2: AI-powered Street View analysis
        ai_analysis = analyze_pickup_spot_with_gemini(lat, lng)
        
        # Stage 3: Nearby POI analysis
        nearby_pois = get_nearby_safety_pois(lat, lng, radius)
        
        # Stage 4: Combine analyses
        safety_score = ai_analysis.get('safety_score', 0.7)
        
        # Penalize if not accessible by car
        if not accessibility_check['is_accessible']:
            safety_score *= 0.3  # Significant penalty for inaccessible spots
            ai_analysis['issues_found'] = ai_analysis.get('issues_found', []) + ['Not accessible by car']
        
        # Adjust score based on nearby POIs
        poi_bonus = min(0.1, len(nearby_pois) * 0.02)  # Small bonus for nearby safety POIs
        final_safety_score = min(1.0, safety_score + poi_bonus)
        
        # Generate additional recommendations based on POIs
        recommendations = ai_analysis.get('recommendations', [])
        
        # Add accessibility warnings
        if not accessibility_check['is_accessible']:
            recommendations.append("🚫 This location may not be accessible by car - consider alternative pickup spot")
        
        if nearby_pois and len(nearby_pois) > 0:
            try:
                closest_poi = min(nearby_pois, key=lambda x: x['distance'])
                if closest_poi['distance'] < 100 and closest_poi['distance'] != float('inf'):
                    recommendations.append(f"✅ {closest_poi['name']} nearby ({closest_poi['distance']:.0f}m)")
            except (ValueError, KeyError) as e:
                print(f"Error processing closest POI: {e}")
                pass
        
        return {
            'is_safe': final_safety_score > 0.6 and accessibility_check['is_accessible'],
            'safety_score': final_safety_score,
            'is_accessible_by_car': accessibility_check['is_accessible'],
            'accessibility_warnings': accessibility_check['warnings'],
            'issues_found': ai_analysis.get('issues_found', []),
            'safety_features': ai_analysis.get('safety_features', []),
            'nearby_pois': nearby_pois,
            'lighting_quality': ai_analysis.get('lighting_quality', 'unknown'),
            'recommendations': recommendations
        }
        
    except Exception as e:
        print(f"Error validating pickup spot: {e}")
        return {
            'is_safe': True,  # Default to safe if validation fails
            'safety_score': 0.7,
            'is_accessible_by_car': True,
            'accessibility_warnings': [],
            'error': str(e),
            'issues_found': [],
            'safety_features': [],
            'nearby_pois': [],
            'lighting_quality': 'unknown',
            'recommendations': ['Validation unavailable - proceed with caution']
        }

# Define routes
@app.route('/api/test', methods=['GET'])
def test_api_status():
    return jsonify({
        "status": "success",
        "message": "API is working!"
    })

@app.route('/api/coordinates', methods=['POST'])
def process_coordinates():
    return jsonify({
        "status": "success",
        "message": "Coordinates received"
    })


@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            try:
                # Get reference location if provided (from the driver's perspective)
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
                print(f"Image saved to {image_path}")
                
                if not os.path.exists(image_path):
                    raise ValueError("Failed to save uploaded file")
                
                # Verify the image can be opened
                test_image = cv2.imread(image_path)
                if test_image is None:
                    raise ValueError("Could not read uploaded image - file may be corrupted")
                
                # Process the image, now with optional reference location for geocoding
                result = process_image(image_path, processed_path, ref_lat, ref_lng)
                
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
                    'image_url': f"/api/image/{filename}"
                }), 200
                
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                # Clean up any partially processed files
                if os.path.exists(image_path):
                    os.remove(image_path)
                if os.path.exists(processed_path):
                    os.remove(processed_path)
                raise
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

def find_relevant_places_for_preferences(feedback, lat, lng, radius=2000):
    """Find relevant places based on user preferences using Google Places API"""
    
    if not feedback or feedback.lower() in ['none', '']:
        return []
    
    # Map common preferences to Google Places types
    preference_mapping = {
        'rental car': ['car_rental'],
        'coffee': ['cafe', 'coffee_shop'],
        'coffee shop': ['cafe', 'coffee_shop'],
        'starbucks': ['cafe'],
        'restaurant': ['restaurant'],
        'food': ['restaurant', 'food'],
        'gas': ['gas_station'],
        'gas station': ['gas_station'],
        'fuel': ['gas_station'],
        'hotel': ['lodging'],
        'motel': ['lodging'],
        'airport': ['airport'],
        'bus': ['bus_station'],
        'train': ['train_station'],
        'subway': ['subway_station'],
        'transit': ['transit_station'],
        'shopping': ['shopping_mall', 'store'],
        'mall': ['shopping_mall'],
        'store': ['store'],
        'pharmacy': ['pharmacy'],
        'drugstore': ['pharmacy'],
        'bank': ['bank'],
        'atm': ['atm'],
        'hospital': ['hospital'],
        'clinic': ['health'],
        'medical': ['health'],
        'police': ['police'],
        'fire': ['fire_station'],
        'park': ['park'],
        'gym': ['gym'],
        'fitness': ['gym'],
        'library': ['library'],
        'school': ['school'],
        'university': ['university'],
        'college': ['university'],
        'post office': ['post_office'],
        'mail': ['post_office'],
        'convenience': ['convenience_store'],
        'convenience store': ['convenience_store'],
        'grocery': ['grocery_or_supermarket'],
        'supermarket': ['grocery_or_supermarket'],
        'market': ['grocery_or_supermarket'],
        'parking': ['parking'],
        'parking lot': ['parking'],
        'lot': ['parking'],
        'garage': ['parking'],
        'well lit': ['streetlight'],
        'lighted': ['streetlight'],
        'bright': ['streetlight'],
        'covered': ['establishment'],
        'roof': ['establishment'],
        'awning': ['establishment'],
        'highway': ['route'],
        'freeway': ['route'],
        'exit': ['route'],
        'ramp': ['route']
    }
    
    # Find matching place types for the feedback
    matching_types = []
    feedback_lower = feedback.lower()
    
    for keyword, place_types in preference_mapping.items():
        if keyword in feedback_lower:
            matching_types.extend(place_types)
    
    # If no specific types found, try a general search
    if not matching_types:
        # Use the feedback as a keyword search
        return search_places_by_keyword(feedback, lat, lng, radius)
    
    # Search for places of the matching types
    found_places = []
    for place_type in matching_types:
        try:
            places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            params = {
                'location': f'{lat},{lng}',
                'radius': radius,
                'type': place_type,
                'key': os.getenv('GOOGLE_MAPS_API_KEY', 'AIzaSyDLwMZDUGQuQh7R3Bg3-b2z8rJ1eYvra8k')
            }
            
            response = requests.get(places_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'OK':
                    for place in data['results']:
                        place_info = {
                            'name': place['name'],
                            'type': place_type,
                            'lat': place['geometry']['location']['lat'],
                            'lng': place['geometry']['location']['lng'],
                            'distance': calculate_distance(lat, lng, place['geometry']['location']['lat'], place['geometry']['location']['lng']),
                            'rating': place.get('rating', 0),
                            'vicinity': place.get('vicinity', '')
                        }
                        found_places.append(place_info)
        except Exception as e:
            print(f"Error searching for {place_type}: {e}")
            continue
    
    return found_places

def search_places_by_keyword(keyword, lat, lng, radius=2000):
    """Search for places using a keyword"""
    
    try:
        places_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            'query': keyword,
            'location': f'{lat},{lng}',
            'radius': radius,
            'key': os.getenv('GOOGLE_MAPS_API_KEY', 'AIzaSyDLwMZDUGQuQh7R3Bg3-b2z8rJ1eYvra8k')
        }
        
        response = requests.get(places_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK':
                found_places = []
                for place in data['results']:
                    place_info = {
                        'name': place['name'],
                        'type': 'keyword_match',
                        'lat': place['geometry']['location']['lat'],
                        'lng': place['geometry']['location']['lng'],
                        'distance': calculate_distance(lat, lng, place['geometry']['location']['lat'], place['geometry']['location']['lng']),
                        'rating': place.get('rating', 0),
                        'vicinity': place.get('formatted_address', '')
                    }
                    found_places.append(place_info)
                return found_places
    except Exception as e:
        print(f"Error searching by keyword: {e}")
    
    return []

def get_llm_pickup_spot(driver, passenger, feedback=None):
    # Find relevant places based on preferences
    relevant_places = []
    if feedback and feedback.lower() not in ['none', '']:
        # Use passenger location as reference for finding relevant places
        relevant_places = find_relevant_places_for_preferences(feedback, passenger['lat'], passenger['lng'])
    
    # Sort places by distance and take the closest ones
    relevant_places.sort(key=lambda x: x['distance'])
    closest_places = relevant_places[:3]  # Take the 3 closest relevant places
    
    # Build the prompt with specific place information
    places_info = ""
    if closest_places:
        places_info = "\n\nRELEVANT PLACES FOUND:\n"
        for i, place in enumerate(closest_places, 1):
            places_info += f"{i}. {place['name']} ({place['type']}) - {place['distance']:.0f}m away\n"
            places_info += f"   Location: {place['lat']}, {place['lng']}\n"
            places_info += f"   Address: {place['vicinity']}\n\n"
    
    prompt = f'''
A driver is at latitude {driver['lat']}, longitude {driver['lng']}, 
and a passenger is at latitude {passenger['lat']}, longitude {passenger['lng']}.

Your task is to find a **realistic pickup location** that:
- Minimizes walking time for the passenger (ideally 2-3 minutes walking).
- Is legally and safely accessible by car.
- Does not require a major detour for the driver.

CRITICAL REQUIREMENTS:
- The pickup spot MUST be on a public road or street where cars can legally stop
- DO NOT suggest pickup spots inside buildings, parking garages, or private property
- DO NOT suggest pickup spots on highways, freeways, or restricted access roads
- The spot must be accessible by a regular car (not requiring special access)
- The spot should be on the street level, not elevated or underground

Valid pickup spot examples:
- Street corners with legal parking
- Public parking areas
- Designated pickup zones
- Street parking spots
- Public lots accessible by car

Invalid pickup spot examples:
- Inside shopping malls
- Inside office buildings
- Underground parking
- Highway shoulders
- Private driveways
- Pedestrian-only areas

Passenger preferences/requirements: "{feedback or 'none'}"

{places_info}

IMPORTANT: If relevant places were found above, choose a pickup spot that is:
1. On the street near one of these places (within 100-200m walking distance)
2. Accessible by car (on a public road where cars can stop)
3. Convenient for both the passenger to walk to and the driver to reach

If no relevant places were found, choose a pickup spot that best matches the general preference while meeting all safety and accessibility requirements.

Return only the decimal latitude and longitude for the new pickup spot.
Example format: 37.7749, -122.4194
'''
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    import re
    text = response.text if hasattr(response, 'text') else str(response)
    match = re.search(r'(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)', text)
    if match:
        lat, lng = float(match.group(1)), float(match.group(2))
        return {'lat': lat, 'lng': lng}, "Pickup point generated by LLM.", closest_places
    else:
        return None, "LLM did not return valid coordinates.", closest_places

@app.route('/api/recommend-pickup', methods=['POST'])
def recommend_pickup():
    """
    Recommend a pickup point using LLM prompt for realistic location. Fallback to driver's location if LLM fails. Do not use midpoint. If passenger walking distance is too far, return error.
    """
    data = request.get_json()
    driver = data.get('driver')
    passenger = data.get('passenger')
    feedback = data.get('feedback', None)
    landmarks = data.get('landmarks', [])
    scene = data.get('scene', '')
    building_type = data.get('building_type', '')

    # Use LLM to get pickup spot with user preferences
    pickup_point, message, relevant_places = get_llm_pickup_spot(driver, passenger, feedback)
    used_llm = pickup_point is not None

    # Fallback: use driver's location if LLM fails
    if not pickup_point:
        pickup_point = {'lat': driver['lat'], 'lng': driver['lng']}
        message = "No valid pickup point found by LLM. Using driver's location as fallback."
        used_llm = False
        relevant_places = []  # No relevant places if LLM failed

    # Validate the pickup spot using AI-powered safety analysis
    if pickup_point and 'lat' in pickup_point and 'lng' in pickup_point:
        validation_result = validate_pickup_spot(pickup_point['lat'], pickup_point['lng'])
    else:
        validation_result = {
            'is_safe': True,
            'safety_score': 0.7,
            'issues_found': [],
            'safety_features': [],
            'nearby_pois': [],
            'lighting_quality': 'unknown',
            'recommendations': ['Pickup point validation unavailable']
        }
    
    # If the pickup spot is unsafe, try to find a safer alternative
    if not validation_result['is_safe'] and used_llm:
        # Try to find a safer spot near the original recommendation
        safer_alternatives = []
        for offset_lat in [-0.001, 0.001, -0.002, 0.002]:  # Try nearby locations
            for offset_lng in [-0.001, 0.001, -0.002, 0.002]:
                alt_lat = pickup_point['lat'] + offset_lat
                alt_lng = pickup_point['lng'] + offset_lng
                alt_validation = validate_pickup_spot(alt_lat, alt_lng)
                if alt_validation['is_safe']:
                    safer_alternatives.append({
                        'lat': alt_lat,
                        'lng': alt_lng,
                        'safety_score': alt_validation['safety_score']
                    })
        
        # Use the safest alternative if found
        if safer_alternatives:
            best_alternative = max(safer_alternatives, key=lambda x: x['safety_score'])
            pickup_point = {'lat': best_alternative['lat'], 'lng': best_alternative['lng']}
            message += f" Safety analysis found a safer pickup spot nearby."
            validation_result = validate_pickup_spot(pickup_point['lat'], pickup_point['lng'])
        else:
            message += f" Safety analysis indicates potential issues with this pickup spot."

    # Directions: (a) passenger to driver, (b) passenger to pickup spot
    directions_to_driver = get_directions_and_street_view_steps(passenger, driver, mode="walking")
    directions_to_pickup = get_directions_and_street_view_steps(passenger, pickup_point, mode="walking")

    if not directions_to_pickup or len(directions_to_pickup) == 0:
        return jsonify({
            'error': 'Could not retrieve directions to pickup point.',
            'message': message
        }), 400

    def enhance_final_step(steps, target_desc):
        if steps and landmarks:
            final_step = steps[-1]
            landmark_descriptions = []
            for lm in landmarks:
                desc = None
                if lm.get('type') == 'text' and lm.get('text'):
                    if len(lm.get('text').split()) < 4:
                        desc = f"a sign that says '{lm.get('text')}'"
                elif lm.get('structure_name'):
                    desc = f"a {lm.get('structure_name').lower()}"
                if desc and desc not in landmark_descriptions:
                    landmark_descriptions.append(desc)
            if landmark_descriptions:
                if len(landmark_descriptions) == 1:
                    landmark_text = landmark_descriptions[0]
                else:
                    landmark_text = ", ".join(landmark_descriptions[:-1]) + f", and {landmark_descriptions[-1]}"
                prompt = f"""
                You are an expert navigation assistant providing the final walking instruction to a passenger meeting their ride.\n\nInstruction from Google Maps: \"{final_step['text']}\"\n\n{target_desc} Your driver is waiting near these landmarks: {landmark_text}.\n\nYour Task:\nCombine the Google Maps instruction with the driver's landmark cues into a single, clear, and helpful paragraph. Your audience is the passenger on foot. Be encouraging and precise. Do not give multiple options. Tell the passenger to look for these specific landmarks to find their driver.
                """
                response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                )
                final_step['text'] = response.text.strip() if hasattr(response, 'text') else final_step['text']
        return steps

    directions_to_driver = enhance_final_step(directions_to_driver, "Destination is the driver.")
    # Do NOT enhance pickup directions with landmarks - they should be generic navigation
    # directions_to_pickup = enhance_final_step(directions_to_pickup, "Destination is the recommended pickup spot.")

    return jsonify({
        'pickup_point': pickup_point,
        'message': message,
        'used_llm': used_llm,
        'directions_to_driver': directions_to_driver,
        'directions_to_pickup': directions_to_pickup,
        'relevant_places': relevant_places,  # Add the relevant places found
        'safety_validation': {
            'is_safe': validation_result['is_safe'],
            'safety_score': validation_result['safety_score'],
            'issues_found': validation_result['issues_found'],
            'recommendations': validation_result['recommendations'],
            'lighting_quality': validation_result.get('street_view_analysis', {}).get('lighting_quality', 'unknown'),
            'nearby_safety_pois': validation_result.get('nearby_pois', {}),
            'is_accessible_by_car': validation_result.get('is_accessible_by_car', True),
            'safety_features': validation_result.get('street_view_analysis', {}).get('safety_features', []),
            'issues_found': validation_result.get('street_view_analysis', {}).get('issues_found', [])
        }
    })

def process_image(image_path: str, output_path: str, ref_lat: str = None, ref_lng: str = None) -> dict:
    """
    Process an image and detect landmarks, scene, and text. Returns structured results for downstream use.
    If a reference location is provided, it will attempt to geocode text-based landmarks.
    """
    try:
        # 1. Read and process the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
        height, width = image.shape[:2]

        # 2. Scene classification
        scene_label = classify_scene(image)

        # 3. Building mask and type
        building_mask_b64 = get_building_mask(image)
        building_mask = np.array(Image.open(io.BytesIO(base64.b64decode(building_mask_b64))))
        mask_area = np.sum(building_mask > 0)
        total_area = building_mask.shape[0] * building_mask.shape[1]
        building_type = 'building'
        if mask_area / total_area > 0.3 and scene_label:
            building_type = scene_label
        box = get_largest_building_box(building_mask)
        building_box = None
        if box:
            x1, y1, x2, y2 = box
            building_box = [float(x1/width), float(y1/height), float(x2/width), float(y2/height)]
            # Draw bounding box and label on image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, building_type, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 4. YOLOv8 object detection
        results = yolov8_model(image)
        structure_landmarks = []
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                class_id = int(cls)
                x1, y1, x2, y2 = box
                width_box = x2 - x1
                height_box = y2 - y1
                aspect_ratio = width_box / height_box if height_box > 0 else 0
                area = width_box * height_box
                label = None
                # Heuristics for buildings
                if class_id == 70:  # 'building' in COCO (not present in default YOLOv8)
                    if area > 0.2 * width * height and aspect_ratio < 1.5:
                        label = 'skyscraper'
                    elif area > 0.1 * width * height and 0.7 < aspect_ratio < 1.3:
                        label = 'dome'
                    else:
                        label = 'building'
                elif class_id == 87:  # 'crane'
                    label = 'crane'
                elif class_id in NOTABLE_STRUCTURES:
                    label = NOTABLE_STRUCTURES[class_id]
                if label:
                    structure_landmarks.append({
                        'type': label,
                        'structure_name': label,
                        'class_id': class_id,
                        'confidence': float(conf),
                        'box': [float(x1/width), float(y1/height), float(x2/width), float(y2/height)]
                    })
        # Fallback: if no structures detected, use contour detection
        if not structure_landmarks:
            structure_landmarks = find_buildings_by_contour(image)

        # 5. EasyOCR for text (global and region-based)
        text_landmarks = detect_text_and_landmarks(image)
        
        # Geocode text landmarks if location is available
        if ref_lat and ref_lng:
            for landmark in text_landmarks:
                if landmark.get('type') == 'text' and landmark.get('text'):
                    try:
                        # Use a helper function to call Google Places API
                        geo_result = geocode_landmark_with_google(landmark['text'], ref_lat, ref_lng)
                        if geo_result:
                            landmark['lat'] = geo_result['lat']
                            landmark['lng'] = geo_result['lng']
                            landmark['address'] = geo_result.get('address', '')
                    except Exception as e:
                        print(f"Could not geocode landmark '{landmark['text']}': {e}")


        # 6. (Stub) GPS/Street View matching
        # Placeholder for future integration
        gps_streetview_matches = []  # TODO: Integrate GPS/Street View matching here

        # 7. Aggregate all landmarks
        landmarks = []
        # Add text landmarks
        for landmark in text_landmarks:
            landmark_data = {
                'type': 'text',
                'text': landmark['text'],
                'category': landmark['category'],
                'confidence': landmark['confidence'],
                'box': landmark['box'],
                'description': f"Text found: {landmark['text']}"
            }
            landmarks.append(landmark_data)
            # Draw on image
            x1 = int(landmark['box'][0] * width)
            y1 = int(landmark['box'][1] * height)
            x2 = int(landmark['box'][2] * width)
            y2 = int(landmark['box'][3] * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, landmark['text'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Add structure landmarks (with region-based OCR for signs)
        for landmark in structure_landmarks:
            landmark_data = process_landmark(landmark, image)
            sign_text = None
            if landmark_data['class_id'] in [9, 11, 91]:  # Traffic Light, Stop Sign, Crosswalk Sign
                x1 = int(landmark_data['box'][0] * width)
                y1 = int(landmark_data['box'][1] * height)
                x2 = int(landmark_data['box'][2] * width)
                y2 = int(landmark_data['box'][3] * height)
                roi = image[y1:y2, x1:x2]
                if roi.size > 0:
                    ocr_results = reader.readtext(roi)
                    if ocr_results:
                        sign_text = ' '.join([text for _, text, prob in ocr_results if prob > 0.5])
            landmark_data['text'] = sign_text
            landmarks.append(landmark_data)
            # Draw on image
            x1 = int(landmark_data['box'][0] * width)
            y1 = int(landmark_data['box'][1] * height)
            x2 = int(landmark_data['box'][2] * width)
            y2 = int(landmark_data['box'][3] * height)
            if landmark_data['class_id'] in [2, 3, 5, 6, 7]:
                color = (255, 0, 0)
            elif landmark_data['class_id'] in [9, 11, 91]:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, landmark_data['structure_name'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Add crosswalk sign landmarks
        crosswalk_sign_landmarks = detect_crosswalk_signs(image)
        for landmark in crosswalk_sign_landmarks:
            landmarks.append(landmark)
            x1 = int(landmark['box'][0] * width)
            y1 = int(landmark['box'][1] * height)
            x2 = int(landmark['box'][2] * width)
            y2 = int(landmark['box'][3] * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(image, landmark['structure_name'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Prepare Gemini prompt for building type (optional, can be removed if not needed)
        detected_objects = [l['structure_name'] for l in landmarks if l.get('structure_name')]
        ocr_texts = [l['text'] for l in landmarks if l['type'] == 'text']
        ocr_text = ", ".join(ocr_texts)
        geolocation = None  # Add geolocation if available in the future
        gemini_type = get_building_type_with_gemini(
            objects=detected_objects,
            scene=scene_label,
            ocr_text=ocr_text,
            mask_area=mask_area / total_area,
            geolocation=geolocation
        )
        # Save the processed image
        cv2.imwrite(output_path, image)
        gemini_type = get_building_type_with_gemini_image(output_path)
        # Return structured results
        result = {
            'landmarks': landmarks,
            'scene': scene_label,
            'building_mask': building_mask_b64,
            'building_type': building_type,
            'building_box': building_box,
            'building_type_gemini': gemini_type,
            'gps_streetview_matches': gps_streetview_matches  # Placeholder for future integration
        }
        return result
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

def geocode_landmark_with_google(query, lat, lng, radius=500):
    """
    Geocode a landmark query using Google Places API Text Search, biased by location.
    """
    try:
        url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={query}&location={lat},{lng}&radius={radius}&key={GOOGLE_MAPS_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'OK' and len(data['results']) > 0:
                result = data['results'][0]
                return {
                    "lat": result['geometry']['location']['lat'],
                    "lng": result['geometry']['location']['lng'],
                    "address": result.get('formatted_address', '')
                }
    except Exception as e:
        print(f"Google Places API request failed: {e}")
    return None

# Define helper functions below
def detect_color(image: np.ndarray, box: List[float]) -> str:
    """Detect the dominant color in a region of the image."""
    height, width = image.shape[:2]
    x1 = int(box[0] * width)
    y1 = int(box[1] * height)
    x2 = int(box[2] * width)
    y2 = int(box[3] * height)
    
    # Extract the region of interest
    roi = image[y1:y2, x1:x2]
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculate the average color
    avg_hsv = np.mean(hsv, axis=(0, 1))
    
    # Simple color classification based on HSV values
    h, s, v = avg_hsv
    
    # First check value (brightness)
    if v < 75:
        return 'black'
    elif v > 230 and s < 30:
        return 'white'
    elif s < 30:
        return 'gray'
    
    # Then check hue for colors
    if h < 15 or h > 165:
        return 'red'
    elif 15 <= h < 30:
        return 'orange'
    elif 30 <= h < 45:
        return 'yellow'
    elif 45 <= h < 75:
        return 'green'
    elif 75 <= h < 105:
        return 'blue'
    elif 105 <= h < 135:
        return 'navy'
    elif 135 <= h < 165:
        return 'purple'
    
    return 'unknown'

def detect_crosswalk_signs(image: np.ndarray) -> List[Dict[str, Any]]:
    """Detects crosswalk signs in the image based on color and shape."""
    height, width = image.shape[:2]
    signs = []

    # Convert image to HSV color space for better color filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for yellow color
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) < 100:
            continue

        # Approximate the contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # A crosswalk sign is typically diamond-shaped (a square with 4 vertices)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Check if it's roughly a square
            if 0.8 <= aspect_ratio <= 1.2:
                box = [
                    float(x) / width,
                    float(y) / height,
                    float(x + w) / width,
                    float(y + h) / height
                ]
                
                signs.append({
                    'type': 'structure',
                    'class_id': 91,  # Custom class for Crosswalk Sign
                    'confidence': 0.75, # Assign a fixed confidence
                    'structure_name': 'Crosswalk Sign',
                    'box': box
                })

    return signs

def detect_material(image: np.ndarray, box: List[float]) -> str:
    """Detect the material of a structure based on texture and color analysis."""
    height, width = image.shape[:2]
    x1 = int(box[0] * width)
    y1 = int(box[1] * height)
    x2 = int(box[2] * width)
    y2 = int(box[3] * height)
    
    # Extract the region of interest
    roi = image[y1:y2, x1:x2]
    
    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Calculate texture features
    glcm = cv2.calcHist([gray], [0], None, [256], [0, 256])
    variance = np.var(gray)
    entropy = -np.sum(glcm * np.log2(glcm + 1e-7))
    
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_color = np.mean(hsv, axis=(0, 1))
    
    # Define material characteristics
    materials = {
        'brick': {
            'color_range': ([0, 50, 50], [20, 255, 255]),  # Reddish-brown
            'variance_range': (1000, 5000),
            'entropy_range': (6, 8)
        },
        'concrete': {
            'color_range': ([0, 0, 100], [180, 30, 200]),  # Gray
            'variance_range': (100, 1000),
            'entropy_range': (4, 6)
        },
        'glass': {
            'color_range': ([0, 0, 150], [180, 50, 255]),  # Transparent/reflective
            'variance_range': (0, 500),
            'entropy_range': (2, 4)
        },
        'metal': {
            'color_range': ([0, 0, 150], [180, 30, 255]),  # Silvery
            'variance_range': (0, 300),
            'entropy_range': (1, 3)
        },
        'wood': {
            'color_range': ([10, 50, 50], [30, 255, 200]),  # Brown
            'variance_range': (500, 2000),
            'entropy_range': (5, 7)
        }
    }
    
    # Score each material based on features
    scores = {}
    for material, chars in materials.items():
        color_score = 0
        if (avg_color[0] >= chars['color_range'][0][0] and 
            avg_color[0] <= chars['color_range'][1][0]):
            color_score = 1
            
        variance_score = 0
        if (variance >= chars['variance_range'][0] and 
            variance <= chars['variance_range'][1]):
            variance_score = 1
            
        entropy_score = 0
        if (entropy >= chars['entropy_range'][0] and 
            entropy <= chars['entropy_range'][1]):
            entropy_score = 1
            
        scores[material] = color_score + variance_score + entropy_score
    
    # Return the material with highest score
    if not scores:
        return 'unknown'
    return max(scores.items(), key=lambda x: x[1])[0]

def detect_material_and_color(image: np.ndarray, box: List[float]) -> Tuple[str, str]:
    """Detect both material and color in a region of the image."""
    height, width = image.shape[:2]
    x1 = int(box[0] * width)
    y1 = int(box[1] * height)
    x2 = int(box[2] * width)
    y2 = int(box[3] * height)
    
    # Extract the region of interest
    roi = image[y1:y2, x1:x2]
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Color detection
    avg_hsv = np.mean(hsv, axis=(0, 1))
    
    # Material detection
    variance = np.var(gray)
    glcm = cv2.calcHist([gray], [0], None, [256], [0, 256])
    entropy = -np.sum(glcm * np.log2(glcm + 1e-7))
    
    # Color ranges
    colors = {
        'white': ([0, 0, 200], [180, 30, 255]),
        'black': ([0, 0, 0], [180, 255, 50]),
        'silver': ([0, 0, 150], [180, 30, 200]),
        'gray': ([0, 0, 100], [180, 30, 150]),
        'red': ([0, 70, 50], [10, 255, 255]),
        'blue': ([100, 50, 50], [130, 255, 255]),
        'green': ([45, 50, 50], [75, 255, 255]),
        'yellow': ([20, 50, 50], [40, 255, 255])
    }
    
    # Material characteristics
    materials = {
        'glass': {'var': (0, 500), 'entropy': (2, 4)},
        'metal': {'var': (0, 300), 'entropy': (1, 3)},
        'concrete': {'var': (100, 1000), 'entropy': (4, 6)},
        'brick': {'var': (1000, 5000), 'entropy': (6, 8)},
        'wood': {'var': (500, 2000), 'entropy': (5, 7)}
    }
    
    # Detect color
    color = 'unknown'
    for name, (lower, upper) in colors.items():
        if (avg_hsv[0] >= lower[0] and avg_hsv[0] <= upper[0] and
            avg_hsv[1] >= lower[1] and avg_hsv[1] <= upper[1] and
            avg_hsv[2] >= lower[2] and avg_hsv[2] <= upper[2]):
            color = name
            break
    
    # Detect material
    material = 'unknown'
    for name, ranges in materials.items():
        if (ranges['var'][0] <= variance <= ranges['var'][1] and
            ranges['entropy'][0] <= entropy <= ranges['entropy'][1]):
            material = name
            break
    
    return material, color

def process_landmark(landmark: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
    """Process a single landmark with enhanced detection."""
    class_id = landmark['class_id']
    structure_name = LANDMARK_CLASSES.get(class_id, 'Unknown Structure')
    
    # Initialize the landmark data
    landmark_data = {
        'type': 'structure',
        'class_id': class_id,
        'structure_name': structure_name,
        'confidence': landmark['confidence'],
        'box': landmark['box']
    }
    
    # Add color detection for vehicles
    if class_id in [2, 3, 5, 6, 7]:  # Vehicles
        color = detect_color(image, landmark['box'])
        if color != 'unknown':
            landmark_data['color'] = color
            structure_name = f"{color.capitalize()} {structure_name}"
    
    landmark_data['structure_name'] = structure_name
    return landmark_data

def find_buildings_by_contour(image, min_area_ratio=0.1):
    height, width = image.shape[:2]
    min_area = min_area_ratio * width * height
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    buildings = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        if area > min_area and 0.3 < aspect_ratio < 3.5:
            buildings.append({
                'type': 'building',
                'structure_name': 'building',
                'class_id': -1,
                'confidence': 0.5,
                'box': [float(x/width), float(y/height), float((x+w)/width), float((y+h)/height)]
            })
    return buildings

def get_largest_building_box(building_mask):
    # Find the largest contour in the building mask
    contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, x + w, y + h)

# Places365 scene recognition setup
PLACES365_MODEL_PATH = 'resnet18_places365.pth.tar'
PLACES365_CATEGORIES_PATH = 'categories_places365.txt'

scene_model = None
scene_classes = None

def load_places365():
    global scene_model, scene_classes
    
    # Download the model if it doesn't exist
    model_url = 'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar'
    if not os.path.exists(PLACES365_MODEL_PATH):
        print(f"Places365 model not found. Downloading from {model_url}...")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(PLACES365_MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading Places365 model: {e}")
            # The application can't continue without this model.
            return

    # Check for categories file
    if not os.path.exists(PLACES365_CATEGORIES_PATH):
        print(f"Error: Places365 categories file not found at {PLACES365_CATEGORIES_PATH}")
        return

    scene_model = models.resnet18(num_classes=365)
    checkpoint = torch.load(PLACES365_MODEL_PATH, map_location='cpu')
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    scene_model.load_state_dict(state_dict)
    scene_model.eval()
    with open(PLACES365_CATEGORIES_PATH) as class_file:
        scene_classes = [line.strip().split(' ')[0][3:] for line in class_file]

load_places365()

scene_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_scene(image_np):
    img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    input_img = scene_transform(img).unsqueeze(0)
    with torch.no_grad():
        logit = scene_model(input_img)
        h_x = torch.nn.functional.softmax(logit, 1).squeeze()
        probs, idx = h_x.sort(0, True)
        return scene_classes[idx[0]] if scene_classes else None

# Load DeepLabV3 segmentation model
segmentation_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# COCO class index for 'building' in DeepLabV3 is 11 (for ADE20K, it's 12)
BUILDING_CLASS_IDX = 11

# Function to get building mask as a base64 PNG
def get_building_mask(image_np):
    input_tensor = scene_transform(Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    with torch.no_grad():
        output = segmentation_model(input_tensor)['out'][0]
        mask = output.argmax(0).cpu().numpy()
    # Create a binary mask for buildings
    building_mask = (mask == BUILDING_CLASS_IDX).astype(np.uint8) * 255
    # Convert mask to PNG base64
    pil_mask = Image.fromarray(building_mask)
    buf = io.BytesIO()
    pil_mask.save(buf, format='PNG')
    mask_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return mask_b64

def get_building_type_with_gemini_image(image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            "What type of building is in this image? Give a short, specific answer."
        ]
    )
    return response.text.strip() if hasattr(response, 'text') else str(response)

def bearing(lat1, lng1, lat2, lng2):
    from math import radians, atan2, sin, cos, degrees
    dLon = radians(lng2 - lng1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    x = sin(dLon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
    brng = atan2(x, y)
    brng = degrees(brng)
    return (brng + 360) % 360

def get_directions_and_street_view_steps(origin, destination, mode="walking"):
    """
    Get directions from Google Maps API and fetch Street View images for each step.
    """
    directions_url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin['lat']},{origin['lng']}&destination={destination['lat']},{destination['lng']}&mode={mode}&key={GOOGLE_MAPS_API_KEY}"
    
    try:
        response = requests.get(directions_url)
        if not response.ok:
            print(f"Error fetching directions: {response.text}")
            return []

        directions_data = response.json()
        if directions_data['status'] != 'OK' or not directions_data.get('routes'):
            print(f"No routes found: {directions_data.get('status')}")
            return []

        steps = []
        route_steps = directions_data['routes'][0]['legs'][0]['steps']
        
        for step in route_steps:
            html_instructions = step['html_instructions']
            start_location = step['start_location']
            end_location = step['end_location']
            
            # Use the start location of the step for the Street View image
            lat, lng = start_location['lat'], start_location['lng']
            
            # Calculate heading for the photo
            heading = bearing(lat, lng, end_location['lat'], end_location['lng'])
            
            # Construct the Street View URL with the heading
            photo_url = f"https://maps.googleapis.com/maps/api/streetview?size=600x400&location={lat},{lng}&heading={heading}&key={GOOGLE_MAPS_API_KEY}"
            
            # Use Gemini to summarize the instruction
            prompt = f"You are a helpful navigation assistant. Rephrase the following instruction for a person who is walking. Be clear, concise, and friendly. Do not offer alternative options, just give one clear instruction. Instruction: '{html_instructions}'"
            
            summary_response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            clean_text = summary_response.text.strip() if hasattr(summary_response, 'text') else html_instructions
            
            steps.append({
                'text': clean_text,
                'photo_url': photo_url
            })
            
        return steps
    except Exception as e:
        print(f"An error occurred while getting directions: {e}")
        return []

GOOGLE_MAPS_API_KEY = "AIzaSyDLwMZDUGQuQh7R3Bg3-b2z8rJ1eYvra8k"

def extract_coordinates(text):
    match = re.search(r"(-?\d+\.\d+)[^\d-]+(-?\d+\.\d+)", text)
    if match:
        try:
            lat = float(match.group(1))
            lng = float(match.group(2))
            if -90 <= lat <= 90 and -180 <= lng <= 180:
                return {'lat': lat, 'lng': lng}
        except:
            return None
    return None

def geocode_address(address):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": GOOGLE_MAPS_API_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data['status'] == 'OK' and data['results']:
            loc = data['results'][0]['geometry']['location']
            return {'lat': loc['lat'], 'lng': loc['lng']}
    except Exception:
        return None
    return None

def ocr_extract_pickup_point(image_url):
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        text = reader.readtext(np.array(img), detail=0)
        text_joined = '\n'.join(text)
        print(f"OCR Text Extracted: {text_joined}")  # Debug

        coords = extract_coordinates(text_joined)
        if coords:
            return coords, "Pickup coordinates extracted from image OCR."

        lines = text_joined.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 5 and any(char.isalpha() for char in line):
                coords = geocode_address(line)
                if coords:
                    return coords, f"Pickup location geocoded from OCR text: '{line}'"

        return None, "No valid pickup location found in OCR text."

    except Exception as e:
        return None, f"OCR failed: {e}"

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
    app.run(debug=True, host='0.0.0.0', port=5002)
