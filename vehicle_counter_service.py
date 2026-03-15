"""
Vehicle Counter Background Service
- Chạy 24/7 đếm xe từ camera IP
- Lưu dữ liệu vào PostgreSQL database
- Có thể chạy như Windows Service hoặc nohup trên Linux
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import time
import psycopg2
from psycopg2 import pool
import logging
from datetime import datetime
from ultralytics import YOLO
import torch
import json
import signal
import sys

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vehicle_counter.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CẤU HÌNH ---
CONFIG = {
    # Camera
    "rtsp_url": "rtsp://admin:02926556833@113.164.79.131:554/cam/realmonitor?channel=1&subtype=0",
    
    # Model
    "model_path": "yolo11n_finetune_v1/weights/best.pt",
    "conf_threshold": 0.7,
    "target_classes": [0, 1, 2, 3],
    "min_frames": 3,
    
    # ROI và LINE (cho resolution 960x720 - không scale)
    "original_width": 960,
    "original_height": 720,
    "roi_points": [(87, 194), (900, 159), (1100, 700), (110, 700)],
    "line_pt1": (407, 719),
    "line_pt2": (767, 323),
    
    # PostgreSQL Database
    "db_host": "localhost",
    "db_port": 5432,
    "db_name": "vehicle_counter",
    "db_user": "postgres",
    "db_password": "postgres",
    
    # Lưu data mỗi bao nhiêu giây
    "save_interval": 60,  # Lưu mỗi phút
    
    # Reconnect settings
    "max_reconnect": 10,
    "reconnect_delay": 5,
}


class VehicleCounterService:
    def __init__(self, config: dict):
        self.config = config
        self.running = False
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Counting state
        self.count_in = 0
        self.count_out = 0
        self.track_history = {}
        self.counted_ids = set()
        self.stable_frames = {}
        
        # Session tracking
        self.session_start = None
        self.last_save_time = None
        self.last_count_in = 0
        self.last_count_out = 0
        
        # Database connection pool
        self.db_pool = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Initialize database
        self.init_database()
        
        logger.info(f"Service initialized. Device: {self.device}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal. Stopping...")
        self.running = False
    
    def get_db_connection(self):
        """Lấy connection từ pool"""
        return self.db_pool.getconn()
    
    def release_db_connection(self, conn):
        """Trả connection về pool"""
        self.db_pool.putconn(conn)
    
    def init_database(self):
        """Khởi tạo PostgreSQL database"""
        try:
            # Tạo connection pool
            self.db_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                host=self.config['db_host'],
                port=self.config['db_port'],
                database=self.config['db_name'],
                user=self.config['db_user'],
                password=self.config['db_password']
            )
            
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Bảng lưu dữ liệu theo phút
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicle_counts (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    count_in INTEGER,
                    count_out INTEGER,
                    total INTEGER,
                    fps REAL,
                    session_id VARCHAR(50)
                )
            ''')
            
            # Bảng lưu sessions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id VARCHAR(50) PRIMARY KEY,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_in INTEGER,
                    total_out INTEGER,
                    status VARCHAR(20)
                )
            ''')
            
            # Bảng lưu events (mỗi xe đi qua)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicle_events (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    track_id INTEGER,
                    direction VARCHAR(10),
                    session_id VARCHAR(50)
                )
            ''')
            
            # Tạo indexes để query nhanh hơn
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_vehicle_counts_timestamp 
                ON vehicle_counts(timestamp)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_vehicle_events_timestamp 
                ON vehicle_events(timestamp)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_vehicle_counts_session 
                ON vehicle_counts(session_id)
            ''')
            
            conn.commit()
            self.release_db_connection(conn)
            
            logger.info(f"PostgreSQL Database initialized: {self.config['db_host']}:{self.config['db_port']}/{self.config['db_name']}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def save_count_data(self, fps: float):
        """Lưu dữ liệu đếm vào database"""
        # Tính số xe mới kể từ lần lưu trước
        new_in = int(self.count_in - self.last_count_in)
        new_out = int(self.count_out - self.last_count_out)
        
        if new_in > 0 or new_out > 0 or True:  # Luôn lưu để có dữ liệu liên tục
            try:
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                # Convert to Python native types
                cursor.execute('''
                    INSERT INTO vehicle_counts (timestamp, count_in, count_out, total, fps, session_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (datetime.now(), new_in, new_out, new_in + new_out, float(fps), self.session_id))
                
                conn.commit()
                self.release_db_connection(conn)
                
                logger.info(f"Saved: +{new_in} IN, +{new_out} OUT | Total session: {self.count_in} IN, {self.count_out} OUT")
            except Exception as e:
                logger.error(f"Error saving count data: {e}")
        
        self.last_count_in = self.count_in
        self.last_count_out = self.count_out
        self.last_save_time = time.time()
    
    def save_vehicle_event(self, track_id: int, direction: str):
        """Lưu event khi xe đi qua line"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Convert numpy int to Python int
            track_id_py = int(track_id)
            
            cursor.execute('''
                INSERT INTO vehicle_events (timestamp, track_id, direction, session_id)
                VALUES (%s, %s, %s, %s)
            ''', (datetime.now(), track_id_py, direction, self.session_id))
            
            conn.commit()
            self.release_db_connection(conn)
        except Exception as e:
            logger.error(f"Error saving vehicle event: {e}")
    
    def start_session(self):
        """Bắt đầu session mới"""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sessions (id, start_time, status)
                VALUES (%s, %s, %s)
            ''', (self.session_id, self.session_start, 'running'))
            
            conn.commit()
            self.release_db_connection(conn)
            
            logger.info(f"Started new session: {self.session_id}")
        except Exception as e:
            logger.error(f"Error starting session: {e}")
    
    def end_session(self):
        """Kết thúc session"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Convert to Python native types
            cursor.execute('''
                UPDATE sessions 
                SET end_time = %s, total_in = %s, total_out = %s, status = %s
                WHERE id = %s
            ''', (datetime.now(), int(self.count_in), int(self.count_out), 'completed', self.session_id))
            
            conn.commit()
            self.release_db_connection(conn)
            
            logger.info(f"Ended session: {self.session_id} | Total: {self.count_in} IN, {self.count_out} OUT")
        except Exception as e:
            logger.error(f"Error ending session: {e}")
        
        # Close connection pool
        if self.db_pool:
            self.db_pool.closeall()
    
    def side_of_line(self, point, line_pt1, line_pt2):
        """Xác định vị trí điểm so với đường thẳng"""
        px, py = point
        x1, y1 = line_pt1
        x2, y2 = line_pt2
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        return cross
    
    def create_video_capture(self):
        """Tạo VideoCapture với cấu hình tối ưu"""
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|framedrop;1"
        os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
        
        cap = cv2.VideoCapture(self.config['rtsp_url'], cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 60000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return cap
    
    def scale_coordinates(self, actual_w: int, actual_h: int):
        """Scale ROI và LINE theo resolution thực tế"""
        orig_w = self.config['original_width']
        orig_h = self.config['original_height']
        
        scale_x = actual_w / orig_w
        scale_y = actual_h / orig_h
        
        logger.info(f"Scaling: {orig_w}x{orig_h} -> {actual_w}x{actual_h} (scale: {scale_x:.2f}x, {scale_y:.2f}y)")
        
        # Scale ROI
        roi_points = []
        for x, y in self.config['roi_points']:
            roi_points.append((int(x * scale_x), int(y * scale_y)))
        
        # Scale LINE
        line_pt1 = (int(self.config['line_pt1'][0] * scale_x), 
                    int(self.config['line_pt1'][1] * scale_y))
        line_pt2 = (int(self.config['line_pt2'][0] * scale_x), 
                    int(self.config['line_pt2'][1] * scale_y))
        
        return roi_points, line_pt1, line_pt2
    
    def save_preview_frame(self, frame, roi_array, line_pt1, line_pt2):
        """Lưu frame preview với ROI và LINE để kiểm tra"""
        preview = frame.copy()
        
        # Vẽ ROI
        cv2.polylines(preview, [roi_array], True, (255, 200, 0), 3)
        cv2.putText(preview, "ROI", (roi_array[0][0] + 10, roi_array[0][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
        
        # Vẽ LINE
        cv2.line(preview, line_pt1, line_pt2, (0, 255, 255), 3)
        cv2.putText(preview, "LINE", (line_pt1[0] + 10, line_pt1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Vẽ thông tin
        height, width = frame.shape[:2]
        info_text = f"Resolution: {width}x{height}"
        cv2.putText(preview, info_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(preview, f"Session: {self.session_id}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Lưu file
        preview_path = f"preview_{self.session_id}.jpg"
        cv2.imwrite(preview_path, preview)
        logger.info(f"📸 Preview saved: {preview_path}")
        
        return preview_path
    
    def save_vehicle_image(self, frame, bbox, track_id, direction, roi_array, line_pt1, line_pt2):
        """Lưu hình ảnh khi xe đi qua line"""
        # Tạo thư mục lưu ảnh nếu chưa có
        img_dir = f"vehicle_images/{self.session_id}"
        os.makedirs(img_dir, exist_ok=True)
        
        # Copy frame
        img = frame.copy()
        
        # Vẽ ROI (mờ hơn)
        cv2.polylines(img, [roi_array], True, (255, 200, 0), 1)
        
        # Vẽ LINE
        cv2.line(img, line_pt1, line_pt2, (0, 255, 255), 2)
        
        # Vẽ bbox xe - màu khác nhau cho IN/OUT
        x1, y1, x2, y2 = bbox
        if direction == 'IN':
            color = (0, 255, 0)  # Xanh lá cho VÀO
            direction_text = "VÀO"
        else:
            color = (0, 0, 255)  # Đỏ cho RA
            direction_text = "RA"
        
        # Vẽ bbox với viền dày
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        # Vẽ label background
        label = f"ID:{track_id} - {direction_text}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Vẽ timestamp và thống kê
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, timestamp_str, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Total: IN={self.count_in} OUT={self.count_out}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Tên file
        timestamp_file = datetime.now().strftime("%H%M%S_%f")[:-3]
        filename = f"{img_dir}/{direction}_{track_id}_{timestamp_file}.jpg"
        
        # Lưu ảnh
        cv2.imwrite(filename, img)
        logger.info(f"📸 Vehicle image saved: {filename}")
    
    def run(self):
        """Main loop - chạy liên tục"""
        self.running = True
        self.start_session()
        
        # Load model
        logger.info("Loading YOLO model...")
        try:
            self.model = YOLO(self.config['model_path'])
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
        
        reconnect_count = 0
        
        while self.running:
            try:
                # Kết nối camera
                logger.info("Connecting to camera...")
                cap = self.create_video_capture()
                
                if not cap.isOpened():
                    raise Exception("Cannot open camera")
                
                # Lấy resolution
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Cannot read frame")
                
                height, width = frame.shape[:2]
                logger.info(f"Camera connected: {width}x{height}")
                
                # Scale coordinates
                roi_points, line_pt1, line_pt2 = self.scale_coordinates(width, height)
                roi_array = np.array(roi_points, np.int32)
                
                # Tạo ROI mask
                roi_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(roi_mask, [roi_array], 255)
                roi_mask_color = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
                
                # Lưu preview frame để kiểm tra ROI/LINE
                self.save_preview_frame(frame, roi_array, line_pt1, line_pt2)
                
                reconnect_count = 0
                prev_time = time.time()
                self.last_save_time = time.time()
                error_count = 0
                fps_list = []
                
                logger.info("Starting detection loop...")
                
                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    
                    if not ret or frame is None:
                        error_count += 1
                        if error_count > 30:
                            raise Exception("Too many read errors")
                        continue
                    
                    error_count = 0
                    
                    # FPS
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                    prev_time = curr_time
                    fps_list.append(fps)
                    if len(fps_list) > 100:
                        fps_list.pop(0)
                    avg_fps = sum(fps_list) / len(fps_list)
                    
                    # Mask ROI
                    masked_frame = cv2.bitwise_and(frame, roi_mask_color)
                    
                    # YOLO tracking
                    try:
                        results = self.model.track(
                            masked_frame,
                            persist=True,
                            tracker="bytetrack.yaml",
                            classes=self.config['target_classes'],
                            conf=self.config['conf_threshold'],
                            iou=0.5,
                            device=self.device,
                            verbose=False,
                            imgsz=640
                        )
                    except Exception as e:
                        logger.warning(f"Detection error: {e}")
                        continue
                    
                    # Xử lý detections
                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                        
                        for box, track_id in zip(boxes, track_ids):
                            x1, y1, x2, y2 = map(int, box)
                            center_x = np.clip((x1 + x2) // 2, 0, width - 1)
                            center_y = np.clip(y2, 0, height - 1)
                            
                            if roi_mask[center_y, center_x] == 0:
                                continue
                            
                            self.stable_frames[track_id] = self.stable_frames.get(track_id, 0) + 1
                            if self.stable_frames[track_id] < self.config['min_frames']:
                                continue
                            
                            if track_id not in self.track_history:
                                self.track_history[track_id] = []
                            
                            self.track_history[track_id].append((center_x, center_y))
                            
                            if len(self.track_history[track_id]) >= 2:
                                prev_point = self.track_history[track_id][-2]
                                curr_point = self.track_history[track_id][-1]
                                
                                prev_side = self.side_of_line(prev_point, line_pt1, line_pt2)
                                curr_side = self.side_of_line(curr_point, line_pt1, line_pt2)
                                
                                if track_id not in self.counted_ids:
                                    if prev_side < 0 and curr_side > 0:
                                        self.count_in += 1
                                        self.counted_ids.add(track_id)
                                        self.save_vehicle_event(track_id, 'IN')
                                        # Lưu hình xe đi VÀO
                                        self.save_vehicle_image(frame, (x1, y1, x2, y2), track_id, 'IN', 
                                                               roi_array, line_pt1, line_pt2)
                                        logger.info(f"🟢 VÀO: ID={track_id} | Total IN: {self.count_in}")
                                    
                                    elif prev_side > 0 and curr_side < 0:
                                        self.count_out += 1
                                        self.counted_ids.add(track_id)
                                        self.save_vehicle_event(track_id, 'OUT')
                                        # Lưu hình xe đi RA
                                        self.save_vehicle_image(frame, (x1, y1, x2, y2), track_id, 'OUT',
                                                               roi_array, line_pt1, line_pt2)
                                        logger.info(f"🔴 RA: ID={track_id} | Total OUT: {self.count_out}")
                    
                    # Lưu data định kỳ
                    if time.time() - self.last_save_time >= self.config['save_interval']:
                        self.save_count_data(avg_fps)
                
                cap.release()
                
            except Exception as e:
                logger.error(f"Error: {e}")
                reconnect_count += 1
                
                if reconnect_count > self.config['max_reconnect']:
                    logger.error("Max reconnect attempts reached. Stopping...")
                    break
                
                logger.info(f"Reconnecting in {self.config['reconnect_delay']}s... ({reconnect_count}/{self.config['max_reconnect']})")
                time.sleep(self.config['reconnect_delay'])
        
        # Cleanup
        self.save_count_data(0)
        self.end_session()
        logger.info("Service stopped")


def main():
    """Entry point"""
    logger.info("=" * 50)
    logger.info("Vehicle Counter Service Starting...")
    logger.info("=" * 50)
    
    # Load config từ file nếu có
    config = CONFIG.copy()
    
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
            logger.info("Loaded config from config.json")
    
    # Chạy service
    service = VehicleCounterService(config)
    service.run()


if __name__ == "__main__":
    main()
