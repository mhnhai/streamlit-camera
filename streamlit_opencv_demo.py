"""
Vehicle Counter Demo - Streamlit + OpenCV Window
- Streamlit: Upload video, config, hiển thị kết quả
- OpenCV: Hiển thị video mượt trong cửa sổ riêng
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO
import torch

st.set_page_config(
    page_title="Vehicle Counter - OpenCV Demo",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Demo Đếm Xe - OpenCV Window")
st.markdown("**Video hiển thị mượt trong cửa sổ OpenCV riêng**")
st.markdown("---")


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


def run_detection(video_path: str, model, config: dict, result_placeholder, status_placeholder):
    """Chạy detection và hiển thị trong OpenCV window"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Không thể mở video!")
        return 0, 0
    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ROI Mask
    roi_detect = np.array([
        (config['roi_x1'], config['roi_y1']),
        (config['roi_x2'], config['roi_y2']),
        (config['roi_x3'], config['roi_y3']),
        (config['roi_x4'], config['roi_y4'])
    ], np.int32)
    
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_detect], 255)
    roi_mask_color = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
    
    # Line
    line_pt1 = (config['line_x1'], config['line_y1'])
    line_pt2 = (config['line_x2'], config['line_y2'])
    
    # Counting variables
    count_in = 0
    count_out = 0
    track_history = {}
    counted_ids = set()
    stable_frames = {}
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    prev_time = time.time()
    frame_count = 0
    
    # Tạo OpenCV window
    window_name = "Vehicle Counter - Press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, width), min(720, height))
    
    status_placeholder.info("🎬 Video đang chạy trong cửa sổ OpenCV. Bấm **Q** để dừng.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
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
                classes=config['target_classes'],
                conf=config['conf_threshold'],
                iou=0.5,
                device=device,
                verbose=False,
                imgsz=640
            )
        except Exception as e:
            continue
        
        # Vẽ ROI
        cv2.polylines(frame, [roi_detect], True, (255, 200, 0), 2)
        
        # Vẽ line
        cv2.line(frame, line_pt1, line_pt2, (0, 255, 255), 3)
        
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
                if stable_frames[track_id] < config['min_frames']:
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
                            count_in += 1
                            counted_ids.add(track_id)
                        elif prev_side > 0 and curr_side < 0:
                            count_out += 1
                            counted_ids.add(track_id)
                
                # Vẽ bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Vẽ thông tin
        cv2.putText(frame, f"VAO:{count_in} RA:{count_out} FPS:{fps:.0f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press Q to quit",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hiển thị trong OpenCV window
        cv2.imshow(window_name, frame)
        
        # Cập nhật metrics trên Streamlit mỗi 30 frames
        if frame_count % 30 == 0:
            with result_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🟢 VÀO", count_in)
                c2.metric("🔴 RA", count_out)
                c3.metric("📊 TỔNG", count_in + count_out)
                c4.metric("⚡ FPS", f"{fps:.0f}")
        
        # Bấm Q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    status_placeholder.success(f"✅ Hoàn tất! Đã xử lý {frame_count} frames")
    
    return count_in, count_out


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


# --- SESSION STATE INIT ---
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'video_info' not in st.session_state:
    st.session_state.video_info = None


# --- MAIN UI ---
st.info(f"🖥️ Device: **{'CUDA' if torch.cuda.is_available() else 'CPU'}**")

st.warning("""
⚠️ **Lưu ý**: Video sẽ hiển thị trong **cửa sổ OpenCV riêng** (không phải trên web).
- Cửa sổ sẽ tự động mở khi bấm "Bắt đầu"
- Bấm phím **Q** trong cửa sổ OpenCV để dừng
- Kết quả đếm sẽ hiển thị trên trang web này
""")

# Upload video
st.subheader("📤 Tải Video Lên")
uploaded_file = st.file_uploader(
    "Chọn file video",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Hỗ trợ các định dạng: MP4, AVI, MOV, MKV",
    key="video_uploader"
)

if uploaded_file is not None:
    # Lưu video vào temp file chỉ khi upload mới
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    if st.session_state.video_path is None or st.session_state.get('file_id') != file_id:
        # Video mới được upload
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.getvalue())  # Dùng getvalue() thay vì read()
        tfile.close()
        st.session_state.video_path = tfile.name
        st.session_state.file_id = file_id
        st.session_state.last_results = None  # Reset results khi upload video mới
        
        # Lưu video info
        cap_info = cv2.VideoCapture(st.session_state.video_path)
        st.session_state.video_info = {
            'width': int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap_info.get(cv2.CAP_PROP_FPS)),
            'frames': int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        cap_info.release()
    
    # Preview video
    st.video(uploaded_file)
    
    # Video info
    info = st.session_state.video_info
    st.info(f"📊 **Video:** {info['width']}x{info['height']} | {info['fps']} FPS | {info['frames']} frames")
    
    # Results section
    st.subheader("📊 Kết quả đếm")
    result_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Hiển thị kết quả cũ nếu có
    if st.session_state.last_results is not None:
        count_in, count_out = st.session_state.last_results
        with result_placeholder.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 VÀO", count_in)
            c2.metric("🔴 RA", count_out)
            c3.metric("📊 TỔNG", count_in + count_out)
        status_placeholder.success(f"✅ Kết quả lần chạy trước")
    
    # Buttons
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        start_btn = st.button("🚀 Bắt đầu đếm xe", type="primary", use_container_width=True)
    
    with col_btn2:
        reset_btn = st.button("🔄 Reset kết quả", use_container_width=True)
    
    if reset_btn:
        st.session_state.last_results = None
        st.rerun()
    
    if start_btn:
        with st.spinner("Đang load model..."):
            model = load_model(model_path)
        
        if model is not None:
            st.success(f"✅ Model loaded!")
            
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
            
            count_in, count_out = run_detection(
                st.session_state.video_path, model, config, 
                result_placeholder, status_placeholder
            )
            
            # Lưu kết quả vào session state
            st.session_state.last_results = (count_in, count_out)
            
            # Final results
            st.subheader("🏁 Kết quả cuối cùng")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🟢 VÀO", count_in)
            with col2:
                st.metric("🔴 RA", count_out)
            with col3:
                st.metric("📊 TỔNG", count_in + count_out)
else:
    # Reset state khi không có file
    st.session_state.video_path = None
    st.session_state.last_results = None
    st.session_state.video_info = None
    st.info("👆 Vui lòng tải video lên để bắt đầu")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚗 Vehicle Counter - Streamlit + OpenCV Window | Video mượt 100%</p>
</div>
""", unsafe_allow_html=True)
