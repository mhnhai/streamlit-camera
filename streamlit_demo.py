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
    page_title="Vehicle Counter - YOLO Demo",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Demo Đếm Xe Qua Vạch - YOLO")
st.markdown("---")

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
    
    st.subheader("🎥 Preview Settings")
    preview_mode = st.radio(
        "Chế độ preview",
        options=["fast", "snapshot"],
        index=0,
        format_func=lambda x: "⚡ Nhanh (không preview)" if x == "fast" else "📸 Snapshot (ảnh mỗi 2s)",
        help="Chế độ 'Nhanh' xử lý nhanh nhất, xem video sau. 'Snapshot' hiển thị ảnh tĩnh định kỳ."
    )


def side_of_line(point, line_pt1, line_pt2):
    """
    Xác định vị trí của một điểm so với đường thẳng
    """
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
def load_model(model_path):
    """Load YOLO model với caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Không thể load model: {e}")
        return None


def process_video(video_path, model, config):
    """Xử lý video và đếm xe"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Không thể mở video!")
        return None, 0, 0
    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
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
    stable_frames_dict = {}
    
    # Output video - sử dụng H.264 codec để tương thích với browser
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))
    
    # Fallback nếu avc1 không hoạt động
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Progress bar và display
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_display = st.empty()
    metrics_placeholder = st.empty()
    
    frame_count = 0
    prev_time = time.time()
    fps_list = []
    
    # Preview settings
    preview_mode = config.get('preview_mode', 'fast')
    
    # Tính preview size
    preview_width = min(640, width)
    preview_height = int(height * preview_width / width)
    
    # Snapshot interval (mỗi 2 giây)
    snapshot_interval = 2.0
    last_snapshot_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # FPS calculation
        curr_time = time.time()
        elapsed = curr_time - prev_time
        fps = 1 / elapsed if elapsed > 0 else 0
        prev_time = curr_time
        fps_list.append(fps)
        
        # Mask ROI
        masked_frame = cv2.bitwise_and(frame, roi_mask_color)
        
        # YOLO tracking
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
                stable_frames_dict[track_id] = stable_frames_dict.get(track_id, 0) + 1
                if stable_frames_dict[track_id] < config['min_frames']:
                    continue
                
                # Track history
                if track_id not in track_history:
                    track_history[track_id] = []
                
                track_history[track_id].append((center_x, center_y))
                
                if len(track_history[track_id]) < 2:
                    continue
                
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
        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:]) if fps_list else 0
        cv2.putText(frame, f"VAO:{count_in} RA:{count_out} FPS:{avg_fps:.0f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Frame: {frame_count}/{total_frames} | FPS: {avg_fps:.1f}")
        
        # Hiển thị metrics realtime (cập nhật mỗi 30 frames để giảm overhead)
        if frame_count % 30 == 0 or frame_count == 1:
            with metrics_placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("🟢 VÀO", count_in)
                m2.metric("🔴 RA", count_out)
                m3.metric("📊 Tổng", count_in + count_out)
        
        # Snapshot mode: hiển thị ảnh tĩnh mỗi 2 giây
        if preview_mode == "snapshot":
            current_time = time.time()
            if current_time - last_snapshot_time >= snapshot_interval:
                last_snapshot_time = current_time
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_small = cv2.resize(frame_rgb, (preview_width, preview_height))
                frame_display.image(frame_small, channels="RGB", caption=f"Snapshot @ frame {frame_count}")
    
    cap.release()
    out.release()
    
    progress_bar.progress(1.0)
    avg_fps_final = sum(fps_list) / len(fps_list) if fps_list else 0
    status_text.text(f"✅ Hoàn tất! {frame_count} frames | Avg FPS: {avg_fps_final:.1f}")
    
    # Xóa preview placeholder
    frame_display.empty()
    
    return output_path, count_in, count_out


# --- MAIN UI ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📤 Tải Video Lên")
    uploaded_file = st.file_uploader(
        "Chọn file video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Hỗ trợ các định dạng: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name
        
        st.video(uploaded_file)
        
        # Get video info
        cap_info = cv2.VideoCapture(temp_video_path)
        video_width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(cap_info.get(cv2.CAP_PROP_FPS))
        video_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_info.release()
        
        st.info(f"📊 **Thông tin video:** {video_width}x{video_height} | {video_fps} FPS | {video_frames} frames")

with col_right:
    st.subheader("🎬 Kết quả xử lý")
    result_placeholder = st.empty()

# Process button
if uploaded_file is not None:
    if st.button("🚀 Bắt đầu đếm xe", type="primary", use_container_width=True):
        with st.spinner("Đang load model..."):
            model = load_model(model_path)
        
        if model is not None:
            st.success(f"✅ Model loaded! Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            
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
                'preview_mode': preview_mode,
            }
            
            with col_right:
                output_video, count_in, count_out = process_video(temp_video_path, model, config)
                
                if output_video:
                    st.subheader("📊 Kết quả cuối cùng")
                    
                    col_in, col_out, col_total = st.columns(3)
                    with col_in:
                        st.metric("🟢 VÀO", count_in)
                    with col_out:
                        st.metric("🔴 RA", count_out)
                    with col_total:
                        st.metric("📈 TỔNG", count_in + count_out)
                    
                    # Hiển thị video đã xử lý
                    st.subheader("🎬 Video đã xử lý")
                    st.video(output_video)
                    
                    # Download processed video
                    with open(output_video, 'rb') as f:
                        st.download_button(
                            label="📥 Tải video đã xử lý",
                            data=f,
                            file_name="processed_video.mp4",
                            mime="video/mp4"
                        )
else:
    st.info("👆 Vui lòng tải video lên để bắt đầu")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚗 Vehicle Counter Demo - Sử dụng YOLO + ByteTrack</p>
</div>
""", unsafe_allow_html=True)
