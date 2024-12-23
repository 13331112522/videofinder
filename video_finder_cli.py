import requests
import base64
import json
import cv2
import numpy as np
#from lightrag import LightRAG
from PIL import Image
from io import BytesIO
from IPython.display import display
import os
from zhipuai import ZhipuAI
#from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Tuple
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the video file path
#video_path = os.getenv('VIDEO_PATH', "02_6.mp4")  # Default to "02_6.mp4" if not set
#frame_interval = int(os.getenv('FRAME_INTERVAL', 60))  # Default to 60 if not set

#client = ZhipuAI(api_key=os.getenv('ZHIPU_API_KEY'))

def get_output_filename(video_path):
    """Generate output filename based on video filename"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    return f"{base_name}_frame_descriptions.txt"

def initialize_frame_descriptions(video_path):
    """Initialize frame_descriptions.txt from backup if it doesn't exist"""
    output_file = get_output_filename(video_path)
    backup_file = f"{output_file}.backup"
    
    if not os.path.exists(output_file) and os.path.exists(backup_file):
        print(f"Initializing {output_file} from backup...")
        with open(backup_file, 'r') as src, open(output_file, 'w') as dst:
            dst.write(src.read())
        print(f"Successfully created {output_file} from backup")
        return True
    return os.path.exists(output_file)

def video_parse(video_path, frame_interval=60):
    output_file = get_output_filename(video_path)
    
    # Check if frame descriptions already exist
    if os.path.exists(output_file):
        print("Frame descriptions already exist, skipping video parsing...")
        return True
        
    cap = cv2.VideoCapture(video_path)
    frame_descriptions = []
    i = 0
    #frame_interval = 60
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Total frames: {total_frames}")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            i += 1
            #print(f"Processing {i} in raw")
            if (i % frame_interval) != 0:
                continue
                
            print(f"Processing frame {i}")
            
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                response = client.chat.completions.create(
                    model="glm-4v-flash",
                    messages=[{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": f"""You are a precise video frame analyzer. Please describe the scene in detail in English, following the instructions below:
                            1.output the description in JSON format start with "{" and end with "}", don't include any other text, with frame number {i} with "frame_number".
                            2.focusing on the person by describing the person's clothes, hair, and other details, contained in "person".
                            3.focusing on the car by describing the car's plate number, brand, model, color, and other details, contained in "car".
                            4.focusing on the scene by describing the scene's place and environment, weather, and other details, contained in "scene".
                            5.focusing on the event by describing what happened in the scene, contained in "event"."""
                        }, {
                            "type": "image_url",
                            "image_url": {
                                "url": frame_base64
                            }
                        }]
                    }]
                )
                
                # Debug: Print the raw response
                # print(f"Raw response for frame {i}: {response}")
                
                # Validate JSON response
                content = response.choices[0].message.content
                
                # Clean the content by removing backticks and "json" text
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]  # Remove ```json
                elif content.startswith("```"):
                    content = content[3:]  # Remove ```
                if content.endswith("```"):
                    content = content[:-3]  # Remove trailing ```
                content = content.strip()
                
                # Debug: Print the cleaned content before parsing
                print(f"Cleaned content for frame {i}: {content}")
                
                # Try to parse as JSON to validate
                try:
                    json.loads(content)
                    frame_descriptions.append(content)
                    print(f"Successfully parsed JSON for frame {i}")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for frame {i}: {e}")
                    print(f"Problematic content: {content[:200]}...")
                    continue
                except Exception as e:
                    print(f"Error processing frame {i}: {str(e)}")
                    continue

            except Exception as e:
                print(f"Error processing frame {i}: {str(e)}")
                continue
        else:
            break
    cap.release()

    # Save only if we have descriptions
    if frame_descriptions:
        with open(output_file, 'w') as f:
            for description in frame_descriptions:
                f.write(f"{description}\n")
        # Create backup
        with open(f"{output_file}.backup", 'w') as f:
            for description in frame_descriptions:
                f.write(f"{description}\n")
        print(f"Saved {len(frame_descriptions)} frame descriptions")
        return True
    else:
        print("No frame descriptions were generated")
        return False

def create_semantic_index(json_file_path):
    index = {}
    frame_texts = {}              
    
    if not os.path.exists(json_file_path):
        print(f"Error: File not found: {json_file_path}")
        return None
    
    try:
        # Read the entire file content
        with open(json_file_path, 'r') as f:
            content = f.read()
        
        # Split into individual JSON objects
        json_objects = []
        current_obj = ""
        brace_count = 0
        
        for char in content:
            current_obj += char
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Clean up the JSON object
                    cleaned_obj = current_obj.strip()
                    if cleaned_obj:
                        json_objects.append(cleaned_obj)
                    current_obj = ""
        
        print(f"Found {len(json_objects)} JSON objects")
        
        # Process each JSON object
        for json_str in json_objects:
            try:
                frame_content = json.loads(json_str)
                
                # Skip frames with explicit errors
                if 'error' in frame_content:
                    print(f"Skipping frame {frame_content.get('frame_number', 'unknown')}: {frame_content.get('error')}")
                    continue
                
                frame_num = frame_content.get('frame_number')
                if not frame_num:
                    continue
                
                # Extract text from person field
                person_text = []
                if person := frame_content.get('person'):
                    if isinstance(person, dict):
                        for key, value in person.items():
                            if isinstance(value, list):
                                # Handle list of strings or dictionaries
                                for item in value:
                                    if isinstance(item, dict):
                                        person_text.extend(str(v) for v in item.values() if v)
                                    else:
                                        person_text.append(str(item))
                            elif value and not isinstance(value, bool):
                                person_text.append(str(value))
                
                # Extract text from car field
                car_text = []
                if car := frame_content.get('car'):
                    if isinstance(car, dict):
                        car_text.extend(str(v) for v in car.values() if v and v != "Not visible" and not isinstance(v, bool))
                
                # Extract text from scene field
                scene_text = []
                if scene := frame_content.get('scene'):
                    if isinstance(scene, dict):
                        scene_text.extend(str(v) for v in scene.values() if v and not isinstance(v, bool))
                
                # Add event text
                event_text = frame_content.get('event', '')
                
                # Combine all texts and clean up
                all_texts = person_text + car_text + scene_text + ([event_text] if event_text else [])
                frame_text = ' '.join(filter(None, all_texts))
                frame_text = frame_text.replace('\n', ' ').strip()
                
                if frame_text:
                    frame_texts[frame_num] = frame_text
                    print(f"Successfully processed frame {frame_num}")
                    print(f"Frame text: {frame_text[:100]}...")  # Print first 100 chars for verification
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON object: {e}")
                print(f"Problematic JSON: {json_str[:200]}...")
                continue
            except Exception as e:
                print(f"Error processing frame content: {e}")
                continue
        
        print(f"Total frames processed: {len(frame_texts)}")
        
        # Create embeddings for all frame descriptions
        if frame_texts:
            try:
                texts = list(frame_texts.values())
                response = client.embeddings.create(
                    model="embedding-3",
                    input=texts
                )
                embeddings = [item.embedding for item in response.data]
                
                for i, frame_num in enumerate(frame_texts.keys()):
                    index[frame_num] = embeddings[i]
                
                print(f"Successfully created embeddings for {len(embeddings)} frames")
                return index
            except Exception as e:
                print(f"Error creating embeddings: {e}")
                return None
        
        return None
        
    except Exception as e:
        print(f"Error reading or processing file: {e}")
        return None

def semantic_search(index, query, threshold=0.3):
    """
    Perform semantic search using sentence embeddings
    Args:
        index: Dictionary of frame numbers and their embeddings
        model: Sentence transformer model
        query: Search query
        threshold: Similarity threshold (0-1)
    Returns:
        List of matching frame numbers sorted by similarity
    """
    if not index:
        return []
    
    print(f"\nSearching for: '{query}'")
    
    response = client.embeddings.create(
            model="embedding-3",
            input=[query],
        )
    query_embedding = np.array(response.data[0].embedding)
    
    #Calculate similarities
    results = []
    similarities = []  # Store all similarities for debugging
    
    for frame_num, frame_embedding in index.items():
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1),
            np.array(frame_embedding).reshape(1, -1)
        )[0][0]
        
        similarities.append((frame_num, similarity))
        if similarity >= threshold:
            results.append((frame_num, similarity))
    
    # Print top 5 similarities regardless of threshold
    print("\nTop 5 most similar frames:")
    for frame_num, sim in sorted(similarities, key=lambda x: x[1], reverse=True)[:5]:
        print(f"Frame {frame_num}: {sim:.4f}")
    
    # Sort by similarity score
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def ground_objects_in_frame(frame, query, api_token):
    """
    Perform grounded segmentation on a frame for a given query
    Args:
        frame: CV2 frame/image
        query: Text query to ground in the image
        api_token: DeepDataSpace API token
    Returns:
        Tuple of (modified_frame, bbox) where bbox is [x1, y1, x2, y2] or None
    """
    # Convert frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    headers = {
        "Content-Type": "application/json",
        "Token": api_token
    }
    
    body = {
        "image": f"data:image/jpeg;base64,{frame_base64}",
        "prompts": [{
            "type": "text",
            "text": query,
            "isPositive": True
        }],
        "model_type": "swint"
    }
    
    # Send initial request
    resp = requests.post(
        url='https://api.deepdataspace.com/tasks/grounded_sam',
        json=body,
        headers=headers
    )
    json_resp = resp.json()
    
    if json_resp['code'] != 0:
        print(f"Error initiating grounding task: {json_resp}")
        return frame, None
        
    task_uuid = json_resp["data"]["task_uuid"]
    
    # Poll for results
    while True:
        resp = requests.get(f'https://api.deepdataspace.com/task_statuses/{task_uuid}', headers=headers)
        json_resp = resp.json()
        if json_resp["data"]["status"] not in ["waiting", "running"]:
            break
        time.sleep(1)
    
    if json_resp["data"]["status"] != "success":
        print(f"Grounding task failed: {json_resp}")
        return frame, None
        
    # Process results
    result = json_resp["data"]["result"]
    output_frame = frame.copy()
    bbox = None
    
    for obj in result.get("objects", []):
        # Get bounding box if available
        if obj.get("bbox"):
            bbox = obj["bbox"]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add text label
            score = obj.get("score", 0)
            label = f"{obj['category']} ({score:.2f})"
            cv2.putText(output_frame, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw mask if available
        if obj.get("mask"):
            mask_rle = obj["mask"]
            # Convert RLE to binary mask
            h, w = mask_rle["size"]
            mask = np.zeros((h, w), dtype=np.uint8)
            # You'll need to implement RLE decoding here or use a library that supports it
            # For visualization, you can overlay the mask with alpha blending
            mask_overlay = output_frame.copy()
            mask_overlay[mask > 0] = [0, 0, 255]  # Red color for mask
            output_frame = cv2.addWeighted(output_frame, 0.7, mask_overlay, 0.3, 0)
    
    return output_frame, bbox

def initialize_tracker(tracker_type='KCF'):
    """Initialize OpenCV tracker based on the specified type."""
    if tracker_type == 'CSRT':
        return cv2.legacy.TrackerCSRT_create()  # For OpenCV 4.x
    elif tracker_type == 'KCF':
        return cv2.legacy.TrackerKCF_create()  # For OpenCV 4.x
    elif tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE_create()  # Updated for OpenCV 4.x
    else:
        print("Invalid tracker type specified.")
        return None
    
def start_tracking(frame: np.ndarray, bbox: list, cap: cv2.VideoCapture, window_name: str):
    """
    Start tracking from given frame and bbox
    """
    # Convert bbox from [x1, y1, x2, y2] to (x, y, w, h)
    x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
    w = x2 - x1
    h = y2 - y1

    # Ensure minimum dimensions
    w = max(30, w)
    h = max(30, h)

    # Check if the bounding box is within the frame dimensions
    frame_height, frame_width = frame.shape[:2]
    if x1 < 0 or y1 < 0 or x1 + w > frame_width or y1 + h > frame_height:
        print(f"Bounding box is out of frame bounds: (x: {x1}, y: {y1}, w: {w}, h: {h})")
        return

    # Print debug information
    print(f"Initial bbox: {bbox}")
    print(f"Converted bbox: (x: {x1}, y: {y1}, w: {w}, h: {h})")

    # Create the tracker
    tracker = initialize_tracker('KCF')
    
    # Initialize tracker with the first frame and bbox
    success = tracker.init(frame, (x1, y1, w, h))
    
    if not success:
        print("Tracker initialization failed")
        return

    print("Tracker initialized successfully!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from video capture.")
            break
        
        # Update tracker
        success, bbox = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

def main():
    # Get API credentials from environment variables
    zhipu_api_key = os.getenv('ZHIPU_API_KEY')
    deepdataspace_api_token = os.getenv('DEEPDATASPACE_API_TOKEN')

    if not zhipu_api_key or not deepdataspace_api_token:
        print("Error: Missing API credentials in .env file")
        exit(1)

    # Initialize ZhipuAI client with API key from environment
    global client
    client = ZhipuAI(api_key=zhipu_api_key)
    
    # Define the video file path and frame interval from environment variables
    video_path = os.getenv('VIDEO_PATH', "02_6.mp4")  # Default to "02_6.mp4" if not set
    frame_interval = int(os.getenv('FRAME_INTERVAL', 60))  # Default to 60 if not set

    # Initialize frame descriptions from backup if needed
    if not initialize_frame_descriptions(video_path):
        # Only parse video if we couldn't initialize from backup
        if not video_parse(video_path, frame_interval):
            print("Failed to create frame descriptions")
            exit(1)
    
    # Create semantic index
    output_file = get_output_filename(video_path)
    print("Creating semantic search index...")
    index = create_semantic_index(output_file)
    if not index:
        print("No valid index created. Please check the frame descriptions format.")
        exit(1)
    
    while True:
        query = input("\nEnter search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        results = semantic_search(index, query, threshold=0.3)
        
        if not results:
            print("No matching frames above threshold.")
            continue
            
        print(f"\nFound {len(results)} matches above threshold:")
        for frame_num, similarity in results:
            print(f"Frame {frame_num}: {similarity:.4f} similarity score")
        
        # Display top matching frames with grounding
        cap = cv2.VideoCapture(video_path)
        current_frame_idx = 0
        results_to_show = results[:5]  # Top 5 matches
        window_name = "Search Results"
        current_bbox = None  # Store the current bbox for tracking
        
        while current_frame_idx < len(results_to_show):
            frame_num, _ = results_to_show[current_frame_idx]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num) - 1)
            ret, frame = cap.read()
            
            if ret:
                # Apply grounding to the frame
                grounded_frame, current_bbox = ground_objects_in_frame(frame, query, deepdataspace_api_token)
                
                # Debug print
                print(f"\nFrame {frame_num} bbox:", current_bbox)
                
                # Add navigation instructions to the frame
                instructions = "Press: 'n'-next, 'p'-previous, 't'-track, 'q'-quit"
                cv2.putText(grounded_frame, instructions, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame number and bbox status
                cv2.putText(grounded_frame, f"Frame {frame_num} ({current_frame_idx + 1}/{len(results_to_show)})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                bbox_status = "BBox Found" if current_bbox is not None else "No BBox"
                cv2.putText(grounded_frame, bbox_status, 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if current_bbox is not None else (0, 0, 255), 2)
                
                cv2.imshow(window_name, grounded_frame)
                
                # Handle key presses
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):  # Quit
                    cv2.destroyWindow(window_name)
                    break
                elif key == ord('n'):  # Next frame
                    current_frame_idx = min(current_frame_idx + 1, len(results_to_show) - 1)
                elif key == ord('p'):  # Previous frame
                    current_frame_idx = max(current_frame_idx - 1, 0)
                elif key == ord('t'):  # Start tracking
                    if current_bbox is not None:
                        # Store current position
                        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        # Start tracking
                        start_tracking(frame, current_bbox, cap, window_name)
                        # Restore position after tracking ends
                        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                    else:
                        print("No bounding box available for tracking")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
