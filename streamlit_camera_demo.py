"""
Vehicle Counter Demo - IP Camera RTSP Stream
- Kết nối camera IP qua RTSP
- YOLO detection realtime
- Hiển thị mượt trong OpenCV window
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

st.set_page_config(
    page_title="Vehicle Counter - IP Camera",
    page_icon="📹",
    layout="wide"
)

st.title("📹 Demo Đếm Xe - IP Camera Realtime")
st.markdown("**Kết nối camera IP qua RTSP + YOLO Detection**")
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


def build_rtsp_url(ip: str, port: str, username: str, password: str, stream_path: str, transport: str = "tcp") -> str:
    """Tạo RTSP URL từ thông tin camera"""
    if username and password:
        return f"rtsp://{username}:{password}@{ip}:{port}/{stream_path}"
    else:
        return f"rtsp://{ip}:{port}/{stream_path}"


def scale_coordinates(config: dict, original_w: int, original_h: int, actual_w: int, actual_h: int) -> dict:
    """Scale ROI và LINE coordinates theo tỉ lệ resolution"""
    scale_x = actual_w / original_w
    scale_y = actual_h / original_h
    
    scaled_config = config.copy()
    
    # Scale LINE
    scaled_config['line_x1'] = int(config['line_x1'] * scale_x)
    scaled_config['line_y1'] = int(config['line_y1'] * scale_y)
    scaled_config['line_x2'] = int(config['line_x2'] * scale_x)
    scaled_config['line_y2'] = int(config['line_y2'] * scale_y)
    
    # Scale ROI
    scaled_config['roi_x1'] = int(config['roi_x1'] * scale_x)
    scaled_config['roi_y1'] = int(config['roi_y1'] * scale_y)
    scaled_config['roi_x2'] = int(config['roi_x2'] * scale_x)
    scaled_config['roi_y2'] = int(config['roi_y2'] * scale_y)
    scaled_config['roi_x3'] = int(config['roi_x3'] * scale_x)
    scaled_config['roi_y3'] = int(config['roi_y3'] * scale_y)
    scaled_config['roi_x4'] = int(config['roi_x4'] * scale_x)
    scaled_config['roi_y4'] = int(config['roi_y4'] * scale_y)
    
    return scaled_config


def create_video_capture(rtsp_url: str, use_tcp: bool = True, timeout_ms: int = 60000) -> cv2.VideoCapture:
    """Tạo VideoCapture với cấu hình tối ưu cho RTSP"""
    
    # Set FFMPEG options qua environment variable
    # - rtsp_transport: tcp để ổn định hơn
    # - fflags: nobuffer để giảm delay
    # - flags: low_delay để giảm latency  
    # - framedrop: 1 để bỏ frame lỗi
    # - err_detect: ignore_err để bỏ qua lỗi decode
    if use_tcp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|framedrop;1"
    else:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|framedrop;1"
    
    # Suppress FFMPEG warnings
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    
    # Tạo capture với FFMPEG backend
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # Cấu hình timeout và buffer
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    return cap


def test_camera_connection(rtsp_url: str, use_tcp: bool = True, timeout_ms: int = 60000) -> tuple:
    """Test kết nối camera, trả về (success, message, frame, width, height)"""
    try:
        cap = create_video_capture(rtsp_url, use_tcp, timeout_ms)
        
        if not cap.isOpened():
            cap.release()
            return False, "Không thể mở kết nối RTSP. Kiểm tra lại IP/Port/Credentials.", None, 0, 0
        
        # Thử đọc vài frame
        for i in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                cap.release()
                return True, "Kết nối thành công!", frame, w, h
            time.sleep(0.5)
        
        cap.release()
        return False, "Kết nối được nhưng không nhận được frame. Thử stream path khác.", None, 0, 0
        
    except Exception as e:
        return False, f"Lỗi: {str(e)}", None, 0, 0


def run_camera_detection(rtsp_url: str, model, config: dict, result_placeholder, status_placeholder, use_tcp: bool = True):
    """Chạy detection từ camera và hiển thị trong OpenCV window"""
    
    cap = create_video_capture(rtsp_url, use_tcp, timeout_ms=60000)
    
    if not cap.isOpened():
        st.error("Không thể kết nối camera!")
        return 0, 0
    
    # Lấy thông tin video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    
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
    window_name = "IP Camera - Vehicle Counter (Press Q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, width), min(720, height))
    
    status_placeholder.info("📹 Camera đang stream trong cửa sổ OpenCV. Bấm **Q** để dừng.")
    
    reconnect_count = 0
    max_reconnect = 5
    error_frame_count = 0
    last_good_frame = None
    
    while True:
        ret, frame = cap.read()
        
        # Xử lý frame lỗi hoặc không đọc được
        if not ret or frame is None:
            error_frame_count += 1
            
            # Nếu lỗi liên tục quá nhiều, thử reconnect
            if error_frame_count > 30:
                reconnect_count += 1
                error_frame_count = 0
                
                if reconnect_count > max_reconnect:
                    status_placeholder.error(f"❌ Mất kết nối camera sau {max_reconnect} lần thử reconnect")
                    break
                
                status_placeholder.warning(f"⚠️ Mất kết nối, đang thử lại... ({reconnect_count}/{max_reconnect})")
                cap.release()
                time.sleep(2)
                cap = create_video_capture(rtsp_url, use_tcp, timeout_ms=30000)
            continue
        
        # Kiểm tra frame có hợp lệ không
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            error_frame_count += 1
            continue
        
        # Frame OK - reset counters
        reconnect_count = 0
        error_frame_count = 0
        last_good_frame = frame.copy()
        
        frame_count += 1
        
        # Resize nếu cần (để phù hợp với ROI mask)
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        
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
        cv2.putText(frame, f"LIVE - Frame: {frame_count}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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
    
    status_placeholder.success(f"✅ Đã dừng! Tổng frames: {frame_count}")
    
    return count_in, count_out


# --- SIDEBAR CẤU HÌNH ---
with st.sidebar:
    st.header("📹 Thông tin Camera")
    
    camera_ip = st.text_input("IP Camera", value="113.164.79.131", help="Địa chỉ IP của camera")
    camera_port = st.text_input("Port", value="554", help="Port RTSP (thường là 554)")
    camera_username = st.text_input("Username", value="admin", help="Tên đăng nhập camera")
    camera_password = st.text_input("Password", value="02926556833", type="password", help="Mật khẩu camera")
    
    st.markdown("---")
    st.markdown("**Stream Path phổ biến:**")
    stream_path_options = {
        "Tùy chỉnh": "",
        "Hikvision (Main)": "Streaming/Channels/101",
        "Hikvision (Sub)": "Streaming/Channels/102",
        "Dahua (Main)": "cam/realmonitor?channel=1&subtype=0",
        "Dahua (Sub)": "cam/realmonitor?channel=1&subtype=1",
        "Generic 1": "stream1",
        "Generic 2": "live/ch00_0",
        "ONVIF": "onvif1",
    }
    
    selected_preset = st.selectbox("Chọn preset", options=list(stream_path_options.keys()), index=3)  # Mặc định Dahua Main
    
    if selected_preset == "Tùy chỉnh":
        stream_path = st.text_input("Stream Path", value="stream1", help="Đường dẫn stream của camera")
    else:
        stream_path = stream_path_options[selected_preset]
        st.code(stream_path, language=None)
    
    st.markdown("---")
    st.subheader("🔧 Cấu hình kết nối")
    
    use_tcp = st.checkbox("Sử dụng TCP (khuyên dùng)", value=True, 
                          help="TCP ổn định hơn UDP, ít mất gói tin")
    
    connection_timeout = st.slider("Timeout (giây)", min_value=10, max_value=120, value=60,
                                   help="Thời gian chờ kết nối camera")
    
    st.markdown("---")
    st.subheader("📐 Resolution gốc (ROI được thiết kế cho)")
    st.caption("ROI/LINE mặc định được thiết kế cho video test1.mp4")
    
    original_width = st.number_input("Width gốc", value=960, step=10, help="Width của video gốc mà ROI được thiết kế")
    original_height = st.number_input("Height gốc", value=720, step=10, help="Height của video gốc mà ROI được thiết kế")
    
    auto_scale_roi = st.checkbox("Tự động scale ROI theo camera", value=True,
                                  help="Tự động điều chỉnh ROI khi camera có resolution khác")
    
    st.markdown("---")
    st.header("⚙️ Cấu hình Model")
    
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
st.info(f"🖥️ Device: **{'CUDA' if torch.cuda.is_available() else 'CPU'}**")

# Build RTSP URL
rtsp_url = build_rtsp_url(camera_ip, camera_port, camera_username, camera_password, stream_path)

# Hiển thị URL (ẩn password)
display_url = build_rtsp_url(camera_ip, camera_port, camera_username, "****", stream_path)
st.code(f"RTSP URL: {display_url}", language=None)

st.warning("""
⚠️ **Lưu ý**: 
- Video sẽ hiển thị trong **cửa sổ OpenCV riêng** (mượt hơn)
- Bấm phím **Q** trong cửa sổ OpenCV để dừng
- Đảm bảo camera và máy tính cùng mạng LAN
""")

# Columns for buttons
col1, col2, col3 = st.columns(3)

with col1:
    test_btn = st.button("🔍 Test kết nối", use_container_width=True)

with col2:
    start_btn = st.button("🚀 Bắt đầu đếm xe", type="primary", use_container_width=True)

with col3:
    if st.button("📋 Copy URL", use_container_width=True):
        st.code(rtsp_url)
        st.info("Copy URL ở trên để test với VLC hoặc công cụ khác")

# Results section
st.subheader("📊 Kết quả")
result_placeholder = st.empty()
status_placeholder = st.empty()
preview_placeholder = st.empty()

# Test connection
if test_btn:
    with st.spinner(f"Đang test kết nối camera (timeout: {connection_timeout}s)..."):
        success, message, frame, cam_w, cam_h = test_camera_connection(rtsp_url, use_tcp, connection_timeout * 1000)
    
    if success:
        st.success(f"✅ {message}")
        if frame is not None:
            # Hiển thị preview frame với ROI và LINE
            display_frame = frame.copy()
            
            # Tính scale nếu cần
            if auto_scale_roi and (cam_w != original_width or cam_h != original_height):
                scale_x = cam_w / original_width
                scale_y = cam_h / original_height
                
                # Scale ROI
                roi_points = np.array([
                    (int(roi_x1 * scale_x), int(roi_y1 * scale_y)),
                    (int(roi_x2 * scale_x), int(roi_y2 * scale_y)),
                    (int(roi_x3 * scale_x), int(roi_y3 * scale_y)),
                    (int(roi_x4 * scale_x), int(roi_y4 * scale_y))
                ], np.int32)
                
                # Scale LINE
                scaled_line_pt1 = (int(line_x1 * scale_x), int(line_y1 * scale_y))
                scaled_line_pt2 = (int(line_x2 * scale_x), int(line_y2 * scale_y))
                
                st.warning(f"⚠️ Resolution khác! Camera: {cam_w}x{cam_h} vs Gốc: {original_width}x{original_height}. ROI sẽ được scale tự động.")
            else:
                roi_points = np.array([
                    (roi_x1, roi_y1), (roi_x2, roi_y2),
                    (roi_x3, roi_y3), (roi_x4, roi_y4)
                ], np.int32)
                scaled_line_pt1 = (line_x1, line_y1)
                scaled_line_pt2 = (line_x2, line_y2)
            
            # Vẽ ROI và LINE lên preview
            cv2.polylines(display_frame, [roi_points], True, (255, 200, 0), 2)
            cv2.line(display_frame, scaled_line_pt1, scaled_line_pt2, (0, 255, 255), 3)
            cv2.putText(display_frame, "ROI", (roi_points[0][0], roi_points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.putText(display_frame, "LINE", (scaled_line_pt1[0], scaled_line_pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(frame_rgb, caption=f"Preview với ROI/LINE - Camera: {cam_w}x{cam_h}", use_container_width=True)
            
            st.info(f"📊 **Camera resolution:** {cam_w}x{cam_h} | **Gốc:** {original_width}x{original_height}")
    else:
        st.error(f"❌ {message}")
        st.markdown("""
        **Kiểm tra:**
        - IP và Port có đúng không?
        - Username/Password có đúng không?
        - Camera có bật không?
        - Máy tính và camera có cùng mạng không?
        - Thử các Stream Path khác trong sidebar
        """)

# Start detection
if start_btn:
    # Test connection first
    with st.spinner(f"Đang test kết nối camera (timeout: {connection_timeout}s)..."):
        success, message, _, cam_w, cam_h = test_camera_connection(rtsp_url, use_tcp, connection_timeout * 1000)
    
    if not success:
        st.error(f"❌ Không thể kết nối: {message}")
    else:
        with st.spinner("Đang load model..."):
            model = load_model(model_path)
        
        if model is not None:
            st.success(f"✅ Model loaded! Camera: {cam_w}x{cam_h}")
            
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
            
            # Scale ROI nếu resolution khác
            if auto_scale_roi and cam_w > 0 and cam_h > 0 and (cam_w != original_width or cam_h != original_height):
                config = scale_coordinates(config, original_width, original_height, cam_w, cam_h)
                st.info(f"📐 ROI đã được scale: {original_width}x{original_height} → {cam_w}x{cam_h}")
            
            count_in, count_out = run_camera_detection(
                rtsp_url, model, config,
                result_placeholder, status_placeholder, use_tcp
            )
            
            # Final results
            st.subheader("🏁 Kết quả cuối cùng")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🟢 VÀO", count_in)
            with col2:
                st.metric("🔴 RA", count_out)
            with col3:
                st.metric("📊 TỔNG", count_in + count_out)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📹 Vehicle Counter - IP Camera RTSP + YOLO | Realtime Detection</p>
</div>
""", unsafe_allow_html=True)
