import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
import tempfile
import threading
import time
from flask import Flask, Response
from ultralytics import YOLO
import torch
import socket

st.set_page_config(
    page_title="Vehicle Counter - MJPEG Stream",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Demo Đếm Xe - MJPEG Stream (Mượt)")
st.markdown("**Video stream mượt với MJPEG + YOLO Detection**")
st.markdown("---")


# Global variables for sharing between Flask and Streamlit
class StreamState:
    def __init__(self):
        self.frame = None
        self.is_running = False
        self.count_in = 0
        self.count_out = 0
        self.fps = 0
        self.lock = threading.Lock()
        self.video_path = None
        self.config = {}
        self.should_stop = False
        self.model = None

stream_state = StreamState()


def side_of_line(point, line_pt1, line_pt2):
    """Xác định vị trí của một điểm so với đường thẳng"""
    px, py = point
    x1, y1 = line_pt1
    x2, y2 = line_pt2
    vector_AB_x = x2 - x1
    vector_AB_y = y2 - y1
    vector_AP_x = px - x1
    vector_AP_y = py - y1
    cross_product = (vector_AB_x * vector_AP_y) - (vector_AB_y * vector_AP_x)
    return cross_product


@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model với caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Không thể load model: {e}")
        return None


def process_video_stream():
    """Process video và update frames cho MJPEG stream"""
    global stream_state
    
    if stream_state.video_path is None:
        return
    
    cap = cv2.VideoCapture(stream_state.video_path)
    if not cap.isOpened():
        return
    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / video_fps
    
    config = stream_state.config
    
    # ROI Mask
    roi_detect = np.array([
        (config.get('roi_x1', 87), config.get('roi_y1', 194)),
        (config.get('roi_x2', 900), config.get('roi_y2', 159)),
        (config.get('roi_x3', 1100), config.get('roi_y3', 700)),
        (config.get('roi_x4', 110), config.get('roi_y4', 700))
    ], np.int32)
    
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_detect], 255)
    roi_mask_color = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
    
    # Line
    line_pt1 = (config.get('line_x1', 407), config.get('line_y1', 719))
    line_pt2 = (config.get('line_x2', 767), config.get('line_y2', 323))
    
    # Counting variables
    track_history = {}
    counted_ids = set()
    stable_frames = {}
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = stream_state.model
    if model is None:
        return
    
    prev_time = time.time()
    
    with stream_state.lock:
        stream_state.count_in = 0
        stream_state.count_out = 0
        stream_state.is_running = True
        stream_state.should_stop = False
    
    while cap.isOpened():
        # Check stop signal
        with stream_state.lock:
            if stream_state.should_stop:
                break
        
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            track_history = {}
            counted_ids = set()
            stable_frames = {}
            continue
        
        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # Mask ROI
        masked_frame = cv2.bitwise_and(frame, roi_mask_color)
        
        # YOLO tracking
        try:
            results = model.track(
                masked_frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=config.get('target_classes', [0, 1, 2, 3]),
                conf=config.get('conf_threshold', 0.5),
                iou=0.5,
                device=device,
                verbose=False,
                imgsz=640
            )
        except Exception:
            continue
        
        # Vẽ ROI
        cv2.polylines(frame, [roi_detect], True, (255, 200, 0), 2)
        
        # Vẽ line
        cv2.line(frame, line_pt1, line_pt2, (0, 255, 255), 3)
        
        count_in_local = stream_state.count_in
        count_out_local = stream_state.count_out
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                center_x = np.clip((x1 + x2) // 2, 0, width - 1)
                center_y = np.clip(y2, 0, height - 1)
                
                # Check ROI
                if roi_mask[center_y, center_x] == 0:
                    continue
                
                # Stable frames
                stable_frames[track_id] = stable_frames.get(track_id, 0) + 1
                if stable_frames[track_id] < config.get('min_frames', 3):
                    continue
                
                # Track history
                if track_id not in track_history:
                    track_history[track_id] = []
                
                track_history[track_id].append((center_x, center_y))
                
                if len(track_history[track_id]) >= 2:
                    prev_point = track_history[track_id][-2]
                    curr_point = track_history[track_id][-1]
                    
                    prev_side = side_of_line(prev_point, line_pt1, line_pt2)
                    curr_side = side_of_line(curr_point, line_pt1, line_pt2)
                    
                    if track_id not in counted_ids:
                        if prev_side < 0 and curr_side > 0:
                            count_in_local += 1
                            counted_ids.add(track_id)
                        elif prev_side > 0 and curr_side < 0:
                            count_out_local += 1
                            counted_ids.add(track_id)
                
                # Vẽ bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Vẽ thông tin
        cv2.putText(frame, f"VAO:{count_in_local} RA:{count_out_local} FPS:{fps:.0f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Update shared state
        with stream_state.lock:
            stream_state.frame = frame.copy()
            stream_state.count_in = count_in_local
            stream_state.count_out = count_out_local
            stream_state.fps = fps
        
        # Control frame rate
        elapsed = time.time() - frame_start
        sleep_time = max(0, frame_delay - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    cap.release()
    with stream_state.lock:
        stream_state.is_running = False


# Flask app for MJPEG streaming
flask_app = Flask(__name__)

def generate_mjpeg():
    """Generator cho MJPEG stream"""
    global stream_state
    while True:
        with stream_state.lock:
            frame = stream_state.frame
        
        if frame is not None:
            # Encode frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            # Placeholder frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for video...", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, jpeg = cv2.imencode('.jpg', placeholder)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS max


@flask_app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def find_free_port():
    """Tìm port trống"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def start_flask_server(port):
    """Start Flask server trong thread riêng"""
    flask_app.run(host='127.0.0.1', port=port, threaded=True, use_reloader=False)


# Initialize Flask server
if 'flask_port' not in st.session_state:
    st.session_state.flask_port = find_free_port()
    flask_thread = threading.Thread(target=start_flask_server, args=(st.session_state.flask_port,), daemon=True)
    flask_thread.start()
    time.sleep(1)  # Wait for server to start


# --- SIDEBAR CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    model_path = st.text_input(
        "Đường dẫn Model YOLO",
        value="yolo11n_finetune_v1/weights/best.pt",
        help="Đường dẫn đến file weights của model YOLO"
    )
    
    conf_threshold = st.slider(
        "Ngưỡng Confidence",
        min_value=0.1, max_value=1.0, value=0.5, step=0.05
    )
    
    min_frames = st.slider(
        "Số frame tối thiểu để ổn định",
        min_value=1, max_value=10, value=3
    )
    
    st.subheader("🎯 Target Classes")
    class_0 = st.checkbox("Class 0", value=True)
    class_1 = st.checkbox("Class 1", value=True)
    class_2 = st.checkbox("Class 2", value=True)
    class_3 = st.checkbox("Class 3", value=True)
    
    target_classes = []
    if class_0: target_classes.append(0)
    if class_1: target_classes.append(1)
    if class_2: target_classes.append(2)
    if class_3: target_classes.append(3)
    
    st.subheader("📏 Counting Line")
    col1, col2 = st.columns(2)
    with col1:
        line_x1 = st.number_input("Line X1", value=407, step=10)
        line_y1 = st.number_input("Line Y1", value=719, step=10)
    with col2:
        line_x2 = st.number_input("Line X2", value=767, step=10)
        line_y2 = st.number_input("Line Y2", value=323, step=10)
    
    st.subheader("🔲 ROI Detect (4 điểm)")
    roi_col1, roi_col2 = st.columns(2)
    with roi_col1:
        roi_x1 = st.number_input("ROI X1", value=87, step=10)
        roi_y1 = st.number_input("ROI Y1", value=194, step=10)
        roi_x2 = st.number_input("ROI X2", value=900, step=10)
        roi_y2 = st.number_input("ROI Y2", value=159, step=10)
    with roi_col2:
        roi_x3 = st.number_input("ROI X3", value=1100, step=10)
        roi_y3 = st.number_input("ROI Y3", value=700, step=10)
        roi_x4 = st.number_input("ROI X4", value=110, step=10)
        roi_y4 = st.number_input("ROI Y4", value=700, step=10)


# --- MAIN UI ---
st.info(f"🖥️ Device: **{'CUDA' if torch.cuda.is_available() else 'CPU'}** | Stream Port: **{st.session_state.flask_port}**")

# Upload video
st.subheader("📤 Tải Video Lên")
uploaded_file = st.file_uploader(
    "Chọn file video",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Hỗ trợ các định dạng: MP4, AVI, MOV, MKV"
)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("🎬 Video Stream (MJPEG)")
    
    # MJPEG stream display
    stream_url = f"http://127.0.0.1:{st.session_state.flask_port}/video_feed"
    
    st.markdown(f"""
    <div style="border: 2px solid #333; border-radius: 10px; overflow: hidden; background: #000;">
        <img src="{stream_url}" style="width: 100%; height: auto;" />
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.subheader("📊 Kết quả đếm")
    
    # Metrics placeholders
    metric_in = st.empty()
    metric_out = st.empty()
    metric_total = st.empty()
    metric_fps = st.empty()
    
    # Control buttons
    st.markdown("---")
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("▶️ Bắt đầu", type="primary", use_container_width=True, disabled=uploaded_file is None):
            if uploaded_file is not None:
                # Save uploaded file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                
                # Load model
                stream_state.model = load_model(model_path)
                
                if stream_state.model:
                    # Update config
                    stream_state.config = {
                        'conf_threshold': conf_threshold,
                        'min_frames': min_frames,
                        'target_classes': target_classes,
                        'line_x1': line_x1, 'line_y1': line_y1,
                        'line_x2': line_x2, 'line_y2': line_y2,
                        'roi_x1': roi_x1, 'roi_y1': roi_y1,
                        'roi_x2': roi_x2, 'roi_y2': roi_y2,
                        'roi_x3': roi_x3, 'roi_y3': roi_y3,
                        'roi_x4': roi_x4, 'roi_y4': roi_y4,
                    }
                    stream_state.video_path = tfile.name
                    
                    # Start processing thread
                    process_thread = threading.Thread(target=process_video_stream, daemon=True)
                    process_thread.start()
                    st.success("✅ Đã bắt đầu stream!")
    
    with col_btn2:
        if st.button("⏹️ Dừng", use_container_width=True):
            with stream_state.lock:
                stream_state.should_stop = True
            st.warning("⏹️ Đã dừng stream")

# Auto-refresh metrics
if stream_state.is_running:
    while True:
        with stream_state.lock:
            count_in = stream_state.count_in
            count_out = stream_state.count_out
            fps = stream_state.fps
            is_running = stream_state.is_running
        
        metric_in.metric("🟢 VÀO", count_in)
        metric_out.metric("🔴 RA", count_out)
        metric_total.metric("📊 TỔNG", count_in + count_out)
        metric_fps.metric("⚡ FPS", f"{fps:.1f}")
        
        if not is_running:
            break
        
        time.sleep(0.5)
else:
    metric_in.metric("🟢 VÀO", stream_state.count_in)
    metric_out.metric("🔴 RA", stream_state.count_out)
    metric_total.metric("📊 TỔNG", stream_state.count_in + stream_state.count_out)
    metric_fps.metric("⚡ FPS", f"{stream_state.fps:.1f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚗 Vehicle Counter - MJPEG Stream + YOLO | Video mượt trên Web</p>
</div>
""", unsafe_allow_html=True)
