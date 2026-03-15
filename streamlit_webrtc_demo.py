import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
import tempfile
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO
import torch
from typing import List
import threading

st.set_page_config(
    page_title="Vehicle Counter - WebRTC Demo",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Demo Đếm Xe - WebRTC (Smooth)")
st.markdown("**Video stream mượt với WebRTC + YOLO Detection**")
st.markdown("---")


@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model với caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Không thể load model: {e}")
        return None


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


class VehicleCounterProcessor(VideoProcessorBase):
    """Video processor cho đếm xe với YOLO"""
    
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Config mặc định
        self.conf_threshold = 0.5
        self.target_classes = [0, 1, 2, 3]
        self.min_frames = 3
        
        # Line config
        self.line_pt1 = (407, 719)
        self.line_pt2 = (767, 323)
        
        # ROI config
        self.roi_points = np.array([(87, 194), (900, 159), (1100, 700), (110, 700)], np.int32)
        self.roi_mask = None
        self.roi_mask_color = None
        
        # Counting state
        self.count_in = 0
        self.count_out = 0
        self.track_history = {}
        self.counted_ids = set()
        self.stable_frames = {}
        
        # Thread lock for shared state
        self.lock = threading.Lock()
        
    def update_config(self, config: dict):
        """Cập nhật config từ sidebar"""
        with self.lock:
            self.conf_threshold = config.get('conf_threshold', 0.5)
            self.target_classes = config.get('target_classes', [0, 1, 2, 3])
            self.min_frames = config.get('min_frames', 3)
            self.line_pt1 = (config.get('line_x1', 407), config.get('line_y1', 719))
            self.line_pt2 = (config.get('line_x2', 767), config.get('line_y2', 323))
            self.roi_points = np.array([
                (config.get('roi_x1', 87), config.get('roi_y1', 194)),
                (config.get('roi_x2', 900), config.get('roi_y2', 159)),
                (config.get('roi_x3', 1100), config.get('roi_y3', 700)),
                (config.get('roi_x4', 110), config.get('roi_y4', 700))
            ], np.int32)
            self.roi_mask = None  # Reset để tạo lại
    
    def reset_counting(self):
        """Reset đếm"""
        with self.lock:
            self.count_in = 0
            self.count_out = 0
            self.track_history = {}
            self.counted_ids = set()
            self.stable_frames = {}
    
    def get_counts(self):
        """Lấy số đếm hiện tại"""
        with self.lock:
            return self.count_in, self.count_out
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Xử lý mỗi frame video"""
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]
        
        # Tạo ROI mask nếu chưa có
        if self.roi_mask is None or self.roi_mask.shape[:2] != (height, width):
            self.roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(self.roi_mask, [self.roi_points], 255)
            self.roi_mask_color = cv2.cvtColor(self.roi_mask, cv2.COLOR_GRAY2BGR)
        
        # Load model nếu chưa có
        if self.model is None:
            self.model = load_model(st.session_state.get('model_path', 'yolo11n_finetune_v1/weights/best.pt'))
        
        if self.model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Mask ROI
        masked_frame = cv2.bitwise_and(img, self.roi_mask_color)
        
        # YOLO tracking
        try:
            results = self.model.track(
                masked_frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=self.target_classes,
                conf=self.conf_threshold,
                iou=0.5,
                device=self.device,
                verbose=False,
                imgsz=640
            )
        except Exception as e:
            cv2.putText(img, f"Error: {str(e)[:50]}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Vẽ ROI
        cv2.polylines(img, [self.roi_points], True, (255, 200, 0), 2)
        
        # Vẽ line
        cv2.line(img, self.line_pt1, self.line_pt2, (0, 255, 255), 3)
        
        # Xử lý detections
        with self.lock:
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    center_x = np.clip((x1 + x2) // 2, 0, width - 1)
                    center_y = np.clip(y2, 0, height - 1)
                    
                    # Check ROI
                    if self.roi_mask[center_y, center_x] == 0:
                        continue
                    
                    # Stable frames
                    self.stable_frames[track_id] = self.stable_frames.get(track_id, 0) + 1
                    if self.stable_frames[track_id] < self.min_frames:
                        continue
                    
                    # Track history
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    
                    self.track_history[track_id].append((center_x, center_y))
                    
                    if len(self.track_history[track_id]) >= 2:
                        prev_point = self.track_history[track_id][-2]
                        curr_point = self.track_history[track_id][-1]
                        
                        prev_side = side_of_line(prev_point, self.line_pt1, self.line_pt2)
                        curr_side = side_of_line(curr_point, self.line_pt1, self.line_pt2)
                        
                        if track_id not in self.counted_ids:
                            if prev_side < 0 and curr_side > 0:
                                self.count_in += 1
                                self.counted_ids.add(track_id)
                            elif prev_side > 0 and curr_side < 0:
                                self.count_out += 1
                                self.counted_ids.add(track_id)
                    
                    # Vẽ bbox
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(img, f"ID:{track_id}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Vẽ thông tin
            cv2.putText(img, f"VAO:{self.count_in} RA:{self.count_out}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- SIDEBAR CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    model_path = st.text_input(
        "Đường dẫn Model YOLO",
        value="yolo11n_finetune_v1/weights/best.pt",
        help="Đường dẫn đến file weights của model YOLO"
    )
    st.session_state['model_path'] = model_path
    
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
st.info(f"🖥️ Device: **{'CUDA' if torch.cuda.is_available() else 'CPU'}**")

# Tabs cho các chế độ
tab1, tab2 = st.tabs(["📹 Webcam / Camera", "📁 Upload Video"])

with tab1:
    st.subheader("Stream từ Webcam")
    st.markdown("Bấm **START** để bắt đầu stream webcam với YOLO detection")
    
    # Config dictionary
    config = {
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
    
    # WebRTC streamer
    ctx = webrtc_streamer(
        key="vehicle-counter",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VehicleCounterProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Cập nhật config khi processor ready
    if ctx.video_processor:
        ctx.video_processor.update_config(config)
        
        # Hiển thị counts
        col1, col2, col3 = st.columns(3)
        
        if st.button("🔄 Reset đếm"):
            ctx.video_processor.reset_counting()
        
        # Placeholder cho metrics
        metrics_placeholder = st.empty()
        
        # Auto-refresh counts
        import time
        while ctx.state.playing:
            count_in, count_out = ctx.video_processor.get_counts()
            with metrics_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("🟢 VÀO", count_in)
                c2.metric("🔴 RA", count_out)
                c3.metric("📊 TỔNG", count_in + count_out)
            time.sleep(0.5)

with tab2:
    st.subheader("📤 Upload Video để xử lý")
    st.markdown("Upload video file và xử lý offline (video output sẽ mượt)")
    
    uploaded_file = st.file_uploader(
        "Chọn file video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Hỗ trợ các định dạng: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file:
        st.video(uploaded_file)
        st.info("💡 Để xử lý video upload, hãy dùng file `streamlit_demo.py` (chế độ offline)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚗 Vehicle Counter - WebRTC + YOLO | Stream mượt realtime</p>
</div>
""", unsafe_allow_html=True)
