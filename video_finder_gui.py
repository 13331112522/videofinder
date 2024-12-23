# video_retrieve_ground_tracking.py
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
import gradio as gr
import sys
from gradio import utils

# Define the video file path
video_path = "02_6.mp4"

# Add default API keys at the top of the file
DEFAULT_ZHIPU_API_KEY = "05ee2930163cef3b10b896a7a8df4c73.JBsGfDATjLorDMv0"
DEFAULT_DEEPDATA_API_KEY = "b1bd372e4a4aaad592852fa37a8a3267"

client = ZhipuAI(api_key=DEFAULT_ZHIPU_API_KEY)

# Loop through the video frames
def video_parse(video_path, frame_interval, zhipu_api_key=DEFAULT_ZHIPU_API_KEY, progress=gr.Progress()):
    client = ZhipuAI(api_key=zhipu_api_key)
    cap = cv2.VideoCapture(video_path)
    frame_descriptions = []
    i = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            i += 1
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
                            1.output the description in JSON format start with "{" and end with "}", with frame number {i} with "frame_number".
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
                
                # Get content from response
                content = response.choices[0].message.content
                
                # Remove any markdown formatting if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].strip()
                
                # Try to parse JSON
                try:
                    parsed_json = json.loads(content)
                    frame_descriptions.append(json.dumps(parsed_json, indent=2))
                    print(f"Successfully processed frame {i}")
                    print(json.dumps(parsed_json, indent=2))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for frame {i}: {e}")
                    print(f"Problematic content: {content}")
                    frame_descriptions.append(json.dumps({
                        "frame_number": i,
                        "error": f"Failed to parse JSON: {str(e)}"
                    }))
                
                progress(i / total_frames)
                
            except Exception as e:
                print(f"Error processing frame {i}: {str(e)}")
                frame_descriptions.append(json.dumps({
                    "frame_number": i,
                    "error": f"Processing error: {str(e)}"
                }))
                continue
        else:
            break

    cap.release()
    
    # Save descriptions to file
    if frame_descriptions:
        with open('frame_descriptions.txt', 'w') as f:
            for description in frame_descriptions:
                f.write(f"{description}\n")
        print(f"Saved {len(frame_descriptions)} frame descriptions")
    else:
        print("No frame descriptions were generated")
    
    return frame_descriptions


def create_semantic_index(json_file_path):
    index = {}
    frame_texts = {}              
    
    if not os.path.exists(json_file_path):
        print(f"Error: File not found: {json_file_path}")
        return None
    
    try:
        with open(json_file_path, 'r') as f:
            content = f.read()
            # Split content into individual JSON objects
            json_objects = content.strip().split('\n}\n')
            json_objects = [obj + '}' for obj in json_objects[:-1]] + [json_objects[-1]]
            
        for json_str in json_objects:
            try:
                if not json_str.strip():
                    continue
                    
                frame_content = json.loads(json_str)
                frame_num = frame_content.get('frame_number')
                
                if not frame_num:
                    continue
                
                # Extract text from all fields
                texts = []
                
                # Process person field
                if person := frame_content.get('person'):
                    if isinstance(person, dict):
                        for value in person.values():
                            if isinstance(value, list):
                                for item in value:
                                    if isinstance(item, dict):
                                        texts.extend(str(v) for v in item.values() if v)
                                    else:
                                        texts.append(str(item))
                            elif value:
                                texts.append(str(value))
                
                # Process car field
                if car := frame_content.get('car'):
                    if isinstance(car, list):
                        for car_item in car:
                            if isinstance(car_item, dict):
                                texts.extend(str(v) for v in car_item.values() if v)
                    elif isinstance(car, dict):
                        texts.extend(str(v) for v in car.values() if v)
                
                # Process scene field
                if scene := frame_content.get('scene'):
                    if isinstance(scene, dict):
                        texts.extend(str(v) for v in scene.values() if v)
                
                # Add event text
                if event := frame_content.get('event'):
                    texts.append(str(event))
                
                # Combine all texts
                frame_text = ' '.join(filter(None, texts))
                
                if frame_text:
                    frame_texts[frame_num] = frame_text
                    print(f"Successfully processed frame {frame_num}")
                
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
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    #query_embedding = model.encode(query)
    response = client.embeddings.create(
        model="embedding-3", #填写需要调用的模型编码
        input=[query],
    )
    query_embedding = response.data[0].embedding
    results = []
    for frame_num, embedding in index.items():
        similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        if similarity >= threshold:
            results.append((frame_num, similarity))
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
    
def start_tracking(video_path, frame_num, bbox):
    """Start tracking from the specified frame and bbox"""
    if not video_path:
        return None, "No video provided"
    if not bbox:
        return None, "No bounding box available for tracking"
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video"
        
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()
    if not ret:
        return None, "Could not read frame"
    
    # Initialize tracker
    tracker = initialize_tracker('KCF')
    x1, y1, x2, y2 = map(int, bbox)
    bbox_tuple = (x1, y1, x2-x1, y2-y1)  # Convert to (x, y, w, h)
    success = tracker.init(frame, bbox_tuple)
    
    if not success:
        return None, "Failed to initialize tracker"
    
    frames = []  # Store processed frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update tracker
        success, bbox = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (100, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        frames.append(frame)
    
    cap.release()
    return frames, "Tracking completed"

def quit_app():
    """Quit the application"""
    try:
        # Release any OpenCV windows
        cv2.destroyAllWindows()
        
        # Force stop the Gradio server
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
    except:
        # If the above doesn't work, use a more aggressive exit
        os._exit(0)

def gradio_interface(zhipu_api_key, deepdata_api_key, video_input, text_query, frame_interval, progress=gr.Progress()):
    # Set API keys
    api_token = deepdata_api_key or DEFAULT_DEEPDATA_API_KEY
    zhipu_api_key = zhipu_api_key or DEFAULT_ZHIPU_API_KEY
    
    if not text_query:
        return None, 0, "Please provide a text query."
    
    # Check if frame_descriptions.txt exists
    if not os.path.exists('frame_descriptions.txt'):
        if not video_input:
            return None, 0, "Please provide a video input or ensure frame_descriptions.txt exists."
        
        if not frame_interval:
            return None, 0, "Please provide a frame interval."
            
        # Process video to create frame descriptions
        print("Processing video to create frame descriptions...")
        frame_descriptions = video_parse(video_input, int(frame_interval), zhipu_api_key, progress)
    else:
        print("Found existing frame descriptions, skipping video processing...")
    
    # Create semantic search index
    print("Creating semantic search index...")
    index = create_semantic_index('frame_descriptions.txt')
    if not index:
        return None, 0, "No valid index created. Please check the frame descriptions format."
    
    # Perform semantic search
    print("Performing semantic search...")
    results = semantic_search(index, text_query)
    
    if not results:
        return None, 1, "No matching frames found."
    
    # Get the frame with highest similarity
    frame_num, similarity = results[0]
    print(f"Best match: Frame {frame_num} with similarity {similarity:.2f}")
    
    # Load and process the frame
    cap = cv2.VideoCapture(video_input)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None, 1, f"Could not read frame {frame_num}"
    
    # Ground the objects in the frame
    processed_frame, bbox = ground_objects_in_frame(frame, text_query, api_token)
    
    if bbox is None:
        return frame, 1, f"Found frame {frame_num} but could not ground objects"
    
    return processed_frame, 1, f"Found and grounded objects in frame {frame_num}"

def track_current_object():
    """Start tracking the current object"""
    if not current_video.value or not current_frame.value or not current_bbox.value:
        return None, "Please process a frame first"
    
    cap = cv2.VideoCapture(current_video.value)
    if not cap.isOpened():
        return None, "Could not open video"
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create temporary output file
    temp_output = "temp_tracking_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame.value - 1)
    ret, frame = cap.read()
    if not ret:
        return None, "Could not read frame"
    
    # Initialize tracker
    tracker = initialize_tracker('KCF')
    x1, y1, x2, y2 = map(int, current_bbox.value)
    bbox_tuple = (x1, y1, x2-x1, y2-y1)  # Convert to (x, y, w, h)
    success = tracker.init(frame, bbox_tuple)
    
    if not success:
        return None, "Failed to initialize tracker"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update tracker
        success, bbox = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (100, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    return temp_output, "Tracking completed"

with gr.Blocks() as iface:
    # Add state variables
    current_frame = gr.State(None)
    current_bbox = gr.State(None)
    current_video = gr.State(None)
    
    with gr.Row():
        with gr.Column():
            zhipu_api_key = gr.Textbox(
                label="ZhipuAI API Key",
                type="password",
                value=DEFAULT_ZHIPU_API_KEY
            )
            deepdata_api_key = gr.Textbox(
                label="DeepDataSpace API Key",
                type="password",
                value=DEFAULT_DEEPDATA_API_KEY
            )
            video_input = gr.Video(
                label="Video Input (optional if frame_descriptions.txt exists)"
            )
            text_query = gr.Textbox(label="Text Query")
            frame_interval = gr.Number(
                label="Frame Interval (only used for new video processing)",
                value=20,
                minimum=1
            )
            
            with gr.Row():
                process_button = gr.Button("Start Processing")
                track_button = gr.Button("Track")
                quit_button = gr.Button("Quit")
                
        with gr.Column():
            # Separate outputs for grounding and tracking
            grounding_output = gr.Image(label="Grounding Result")  # Changed from video_output to grounding_output
            tracking_output = gr.Video(label="Tracking Result")    # New video output for tracking
            progress_bar = gr.Slider(label="Progress", minimum=0, maximum=1, value=0)
            status_text = gr.Textbox(label="Status", interactive=False)
    
    def process_and_update_state(zhipu_key, deepdata_key, video, query, interval):
        """Process video and store state for tracking"""
        result = gradio_interface(zhipu_key, deepdata_key, video, query, interval)
        if result:
            frame, progress, status = result
            # Update Gradio state values
            current_video.value = video
            
            # Extract frame number from status
            if isinstance(status, str) and "frame" in status:
                try:
                    frame_num = int(status.split("frame")[1].split()[0])
                    current_frame.value = frame_num
                    
                    # Get the bbox from ground_objects_in_frame
                    cap = cv2.VideoCapture(video)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        _, bbox = ground_objects_in_frame(frame, query, deepdata_key)
                        current_bbox.value = bbox
                        print(f"Stored bbox: {bbox}")  # Debug print
                except Exception as e:
                    print(f"Error capturing bbox: {e}")
                    pass
        return result
    
    def track_current_object():
        """Start tracking the current object"""
        if not current_video.value or not current_frame.value or not current_bbox.value:
            return None, "Please process a frame first"
        
        cap = cv2.VideoCapture(current_video.value)
        if not cap.isOpened():
            return None, "Could not open video"
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create temporary output file
        temp_output = "temp_tracking_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame.value - 1)
        ret, frame = cap.read()
        if not ret:
            return None, "Could not read frame"
        
        # Initialize tracker
        tracker = initialize_tracker('KCF')
        x1, y1, x2, y2 = map(int, current_bbox.value)
        bbox_tuple = (x1, y1, x2-x1, y2-y1)  # Convert to (x, y, w, h)
        success = tracker.init(frame, bbox_tuple)
        
        if not success:
            return None, "Failed to initialize tracker"
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update tracker
            success, bbox = tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Lost", (100, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        return temp_output, "Tracking completed"

    # Connect the buttons
    process_button.click(
        process_and_update_state,
        inputs=[zhipu_api_key, deepdata_api_key, video_input, text_query, frame_interval],
        outputs=[grounding_output, progress_bar, status_text]  # Changed from video_output to grounding_output
    )
    
    track_button.click(
        fn=track_current_object,
        inputs=[],
        outputs=[tracking_output, status_text]  # Output to tracking_output
    )
    
    quit_button.click(
        fn=quit_app,
        inputs=None,
        outputs=None
    )

    # Create a state for tracking
    is_tracking = gr.State(False)
    
    # Add this at the end of your interface definition
    iface.load(lambda: None, None, None, every=0.1)  # Refresh the interface periodically

if __name__ == "__main__":
    iface.queue()  # Enable queueing if needed
    iface.launch(show_error=True)