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

# 加载环境变量
load_dotenv()

# 定义视频文件路径
#video_path = os.getenv('VIDEO_PATH', "02_6.mp4")  # 如果未设置，默认为 "02_6.mp4"
#frame_interval = int(os.getenv('FRAME_INTERVAL', 60))  # 如果未设置，默认为 60

#client = ZhipuAI(api_key=os.getenv('ZHIPU_API_KEY'))

def get_output_filename(video_path):
    """根据视频文件名生成输出文件名"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    return f"{base_name}_frame_descriptions_zh.txt"

def initialize_frame_descriptions(video_path):
    """如果 frame_descriptions.txt 不存在，则从备份初始化"""
    output_file = get_output_filename(video_path)
    backup_file = f"{output_file}.backup"
    
    if not os.path.exists(output_file) and os.path.exists(backup_file):
        print(f"从备份初始化 {output_file}...")
        with open(backup_file, 'r') as src, open(output_file, 'w') as dst:
            dst.write(src.read())
        print(f"成功从备份创建 {output_file}")
        return True
    return os.path.exists(output_file)

def video_parse(video_path, frame_interval=60):
    output_file = get_output_filename(video_path)
    
    # 检查帧描述是否已存在
    if os.path.exists(output_file):
        print("帧描述已存在，跳过视频解析...")
        return True
        
    cap = cv2.VideoCapture(video_path)
    frame_descriptions = []
    i = 0
    #frame_interval = 60
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"总帧数: {total_frames}")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            i += 1
            #print(f"Processing {i} in raw")
            if (i % frame_interval) != 0:
                continue
                
            print(f"处理帧 {i}")
            
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                response = client.chat.completions.create(
                    model="glm-4v-flash",
                    messages=[{
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": f"""你是一个精确的视频帧分析器。请按照以下说明详细描述场景，用英文描述:
                            1.输出以 "{" 开头和 "}" 结尾的 JSON 格式描述，不包含任何其他文本，帧号为 {i}，用 "帧数" 表示。
                            2.关注人物，描述人物的衣服、头发和其他细节，包含在 "行人" 中。
                            3.关注汽车，描述汽车的车牌号、品牌、型号、颜色和其他细节，包含在 "车辆" 中。
                            4.关注场景，描述场景的地点和环境、天气和其他细节，包含在 "场景" 中。
                            5.关注事件，描述场景中发生的事情，包含在 "事件" 中。"""
                        }, {
                            "type": "image_url",
                            "image_url": {
                                "url": frame_base64
                            }
                        }]
                    }]
                )
                
                # 调试: 打印原始响应
                # print(f"Raw response for frame {i}: {response}")
                
                # 验证 JSON 响应
                content = response.choices[0].message.content
                
                # 清理内容，去除反引号和 "json" 文本
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]  # 去除 ```json
                elif content.startswith("```"):
                    content = content[3:]  # 去除 ```
                if content.endswith("```"):
                    content = content[:-3]  # 去除尾部 ```
                content = content.strip()
                
                # 调试: 在解析前打印清理后的内容
                #print(f"清理后的内容，帧 {i}: {content}")
                
                # 尝试解析为 JSON 以验证
                try:
                    json.loads(content)
                    frame_descriptions.append(content)
                    print(f"成功解析 JSON，帧 {i}")
                except json.JSONDecodeError as e:
                    print(f"JSON 解码错误，帧 {i}: {e}")
                    print(f"有问题的内容: {content[:200]}...")
                    continue
                except Exception as e:
                    print(f"处理帧 {i} 时出错: {str(e)}")
                    continue

            except Exception as e:
                print(f"处理帧 {i} 时出错: {str(e)}")
                continue
        else:
            break
    cap.release()

    # 仅在有描述时保存
    if frame_descriptions:
        with open(output_file, 'w') as f:
            for description in frame_descriptions:
                f.write(f"{description}\n")
        # 创建备份
        with open(f"{output_file}.backup", 'w') as f:
            for description in frame_descriptions:
                f.write(f"{description}\n")
        print(f"保存了 {len(frame_descriptions)} 个帧描述")
        return True
    else:
        print("未生成任何帧描述")
        return False

def create_semantic_index(json_file_path):
    index = {}
    frame_texts = {}              
    
    if not os.path.exists(json_file_path):
        print(f"错误: 文件未找到: {json_file_path}")
        return None
    
    try:
        # 读取整个文件内容
        with open(json_file_path, 'r') as f:
            content = f.read()
        
        # 拆分为单个 JSON 对象
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
                    # 清理 JSON 对象
                    cleaned_obj = current_obj.strip()
                    if cleaned_obj:
                        json_objects.append(cleaned_obj)
                    current_obj = ""
        
        print(f"找到 {len(json_objects)} 个 JSON 对象")
        
        # 处理每个 JSON 对象
        for json_str in json_objects:
            try:
                frame_content = json.loads(json_str)
                
                # 跳过有明确错误的帧
                if 'error' in frame_content:
                    print(f"跳过帧 {frame_content.get('帧数', 'unknown')}: {frame_content.get('error')}")
                    continue
                
                frame_num = frame_content.get('帧数')
                if not frame_num:
                    continue
                
                # 从 person 字段提取文本
                person_text = []
                if person := frame_content.get('行人'):
                    if isinstance(person, dict):
                        for key, value in person.items():
                            if isinstance(value, list):
                                # 处理字符串或字典列表
                                for item in value:
                                    if isinstance(item, dict):
                                        person_text.extend(str(v) for v in item.values() if v)
                                    else:
                                        person_text.append(str(item))
                            elif value and not isinstance(value, bool):
                                person_text.append(str(value))
                
                # 从 car 字段提取文本
                car_text = []
                if car := frame_content.get('车辆'):
                    if isinstance(car, dict):
                        car_text.extend(str(v) for v in car.values() if v and v != "Not visible" and not isinstance(v, bool))
                
                # 从 scene 字段提取文本
                scene_text = []
                if scene := frame_content.get('场景'):
                    if isinstance(scene, dict):
                        scene_text.extend(str(v) for v in scene.values() if v and not isinstance(v, bool))
                
                # 添加事件文本
                event_text = frame_content.get('事件', '')
                
                # 合并所有文本并清理
                all_texts = person_text + car_text + scene_text + ([event_text] if event_text else [])
                frame_text = ' '.join(filter(None, all_texts))
                frame_text = frame_text.replace('\n', ' ').strip()
                
                if frame_text:
                    frame_texts[frame_num] = frame_text
                    print(f"成功处理帧 {frame_num}")
                    print(f"帧文本: {frame_text[:100]}...")  # 打印前 100 个字符以供验证
                
            except json.JSONDecodeError as e:
                print(f"解析 JSON 对象时出错: {e}")
                print(f"有问题的 JSON: {json_str[:200]}...")
                continue
            except Exception as e:
                print(f"处理帧内容时出错: {e}")
                continue
        
        print(f"总共处理了 {len(frame_texts)} 帧")
        
        # 为所有帧描述创建嵌入
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
                
                print(f"成功为 {len(embeddings)} 帧创建嵌入")
                return index
            except Exception as e:
                print(f"创建嵌入时出错: {e}")
                return None
        
        return None
        
    except Exception as e:
        print(f"读取或处理文件时出错: {e}")
        return None

def semantic_search(index, query, threshold=0.3):
    """
    使用句子嵌入进行语义搜索
    参数:
        index: 帧号及其嵌入的字典
        model: 句子转换模型
        query: 搜索查询
        threshold: 相似度阈值 (0-1)
    返回:
        按相似度排序的匹配帧号列表
    """
    if not index:
        return []
    
    print(f"\n搜索: '{query}'")
    
    response = client.embeddings.create(
            model="embedding-3",
            input=[query],
        )
    query_embedding = np.array(response.data[0].embedding)
    
    #计算相似度
    results = []
    similarities = []  # 存储所有相似度以供调试
    
    for frame_num, frame_embedding in index.items():
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1),
            np.array(frame_embedding).reshape(1, -1)
        )[0][0]
        
        similarities.append((frame_num, similarity))
        if similarity >= threshold:
            results.append((frame_num, similarity))
    
    # 打印前 5 个相似度最高的帧，无论阈值如何
    print("\n最相似的前 5 帧:")
    for frame_num, sim in sorted(similarities, key=lambda x: x[1], reverse=True)[:5]:
        print(f"帧 {frame_num}: {sim:.4f}")
    
    # 按相似度排序
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def ground_objects_in_frame(frame, query, api_token):
    """
    对帧进行基于查询的分割
    参数:
        frame: CV2 帧/图像
        query: 要在图像中定位的文本查询
        api_token: DeepDataSpace API 令牌
    返回:
        (modified_frame, bbox) 元组，其中 bbox 为 [x1, y1, x2, y2] 或 None
    """
    # 将帧转换为 base64
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
    
    # 发送初始请求
    resp = requests.post(
        url='https://api.deepdataspace.com/tasks/grounded_sam',
        json=body,
        headers=headers
    )
    json_resp = resp.json()
    
    if json_resp['code'] != 0:
        print(f"初始化分割任务时出错: {json_resp}")
        return frame, None
        
    task_uuid = json_resp["data"]["task_uuid"]
    
    # 轮询结果
    while True:
        resp = requests.get(f'https://api.deepdataspace.com/task_statuses/{task_uuid}', headers=headers)
        json_resp = resp.json()
        if json_resp["data"]["status"] not in ["waiting", "running"]:
            break
        time.sleep(1)
    
    if json_resp["data"]["status"] != "success":
        print(f"分割任务失败: {json_resp}")
        return frame, None
        
    # 处理结果
    result = json_resp["data"]["result"]
    output_frame = frame.copy()
    bbox = None
    
    for obj in result.get("objects", []):
        # 获取边界框（如果有）
        if obj.get("bbox"):
            bbox = obj["bbox"]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 添加文本标签
            score = obj.get("score", 0)
            label = f"{obj['category']} ({score:.2f})"
            cv2.putText(output_frame, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制掩码（如果有）
        if obj.get("mask"):
            mask_rle = obj["mask"]
            # 将 RLE 转换为二进制掩码
            h, w = mask_rle["size"]
            mask = np.zeros((h, w), dtype=np.uint8)
            # 需要在这里实现 RLE 解码或使用支持它的库
            # 对于可视化，可以使用 alpha 混合叠加掩码
            mask_overlay = output_frame.copy()
            mask_overlay[mask > 0] = [0, 0, 255]  # 红色掩码
            output_frame = cv2.addWeighted(output_frame, 0.7, mask_overlay, 0.3, 0)
    
    return output_frame, bbox

def initialize_tracker(tracker_type='KCF'):
    """根据指定类型初始化 OpenCV 跟踪器。"""
    if tracker_type == 'CSRT':
        return cv2.legacy.TrackerCSRT_create()  # 适用于 OpenCV 4.x
    elif tracker_type == 'KCF':
        return cv2.legacy.TrackerKCF_create()  # 适用于 OpenCV 4.x
    elif tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE_create()  # 更新为 OpenCV 4.x
    else:
        print("指定的跟踪器类型无效。")
        return None
    
def start_tracking(frame: np.ndarray, bbox: list, cap: cv2.VideoCapture, window_name: str):
    """
    从给定的帧和边界框开始跟踪
    """
    # 将 bbox 从 [x1, y1, x2, y2] 转换为 (x, y, w, h)
    x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
    w = x2 - x1
    h = y2 - y1

    # 确保最小尺寸
    w = max(30, w)
    h = max(30, h)

    # 检查边界框是否在帧尺寸内
    frame_height, frame_width = frame.shape[:2]
    if x1 < 0 or y1 < 0 or x1 + w > frame_width or y1 + h > frame_height:
        print(f"边界框超出帧范围: (x: {x1}, y: {y1}, w: {w}, h: {h})")
        return

    # 打印调试信息
    print(f"初始 bbox: {bbox}")
    print(f"转换后的 bbox: (x: {x1}, y: {y1}, w: {w}, h: {h})")

    # 创建跟踪器
    tracker = initialize_tracker('KCF')
    
    # 使用第一帧和 bbox 初始化跟踪器
    success = tracker.init(frame, (x1, y1, w, h))
    
    if not success:
        print("跟踪器初始化失败")
        return

    print("跟踪器初始化成功！")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("从视频捕获读取帧失败。")
            break
        
        # 更新跟踪器
        success, bbox = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
