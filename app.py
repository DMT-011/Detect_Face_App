import cv2
import streamlit as st
import numpy as np
from PIL import Image
from utils.loader import load_haar_cascade, load_trained_cnn_model
from utils.detection import detect_faces_haar, detect_faces_cnn
from utils.sample_images import load_sample_images


# Cấu hình trang
st.set_page_config(
    page_title="Face Detection Demo - Haar vs CNN",
    page_icon="👤",
    layout="wide"
)

def main():
    st.title("Face Detection Demo: Haar Cascade vs CNN")
    st.markdown("### So sánh phương pháp nhận diện khuôn mặt")
    st.markdown("---")
    
    # Load models and data
    haar_cascade = load_haar_cascade()
    cnn_model, model_file = load_trained_cnn_model()
    
    # Sidebar
    st.sidebar.header("Cài đặt")
    
    # Detection method selection
    detection_options = ["Haar Cascade"]
    
    if cnn_model is not None:
        detection_options.extend(["CNN", "So sánh cả hai"])
        cnn_label = "CNN"
    
    detection_method = st.sidebar.selectbox(
        "Chọn phương pháp detect:",
        detection_options
    )
    
    # CNN confidence threshold
    if detection_method in ["CNN", "So sánh cả hai"]:
        confidence_threshold = st.sidebar.slider(
            "CNN Confidence Threshold:",
            min_value=0.1, max_value=0.9, value=0.7, step=0.1
        )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Chọn ảnh để test")
        
        # Image source selection
        image_source = st.radio(
            "Nguồn ảnh:",
            ["Ảnh mẫu", "Upload ảnh"]
        )
        
        selected_image = None
        
        if image_source == "Ảnh mẫu":
            sample_images = load_sample_images()
            
            image_idx = st.selectbox(
                "Chọn ảnh mẫu:",
                range(len(sample_images)),
                format_func=lambda x: f"Ảnh mẫu {x+1}"
            )
            
            selected_image = sample_images[image_idx]
            st.image(selected_image, caption=f"Ảnh mẫu {image_idx+1}", use_column_width=True)
            
        elif image_source == "Upload ảnh":
            uploaded_file = st.file_uploader(
                "Upload ảnh:",
                type=['jpg', 'jpeg', 'png']
            )
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    selected_image = np.array(image)
                    st.image(image, caption="Ảnh đã upload", use_column_width=True)
                except Exception as e:
                    st.error(f"Lỗi đọc ảnh: {e}")
        
        else:  # Webcam
            st.info("Webcam feature - Tính năng này cần camera và setup thêm")
            
            if st.button("Chụp ảnh giả lập"):
                webcam_image = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
                cv2.rectangle(webcam_image, (100, 80), (200, 200), (120, 120, 120), -1)
                cv2.circle(webcam_image, (130, 120), 10, (80, 80, 80), -1)
                cv2.circle(webcam_image, (170, 120), 10, (80, 80, 80), -1)
                cv2.ellipse(webcam_image, (150, 160), (20, 10), 0, 0, 180, (80, 80, 80), -1)
                
                selected_image = webcam_image
                st.image(webcam_image, caption="Ảnh từ webcam (giả lập)", use_column_width=True)
    
    with col2:
        st.header("Kết quả Detection")
        
        if selected_image is not None:
            if detection_method == "Haar Cascade":
                result_img, face_count, det_time = detect_faces_haar(selected_image, haar_cascade)
                st.image(result_img, caption=f"Haar Cascade - Tìm thấy {face_count} khuôn mặt", use_column_width=True)
                
                col2a, col2b, col2c = st.columns(3)
                with col2a:
                    st.metric("Số mặt phát hiện", face_count)
                with col2b:
                    st.metric("Thời gian", f"{det_time:.3f}s")
                with col2c:
                    st.metric("Tốc độ", "Nhanh" if det_time < 1 else "Chậm")
                
            elif detection_method == "CNN" and cnn_model is not None:
                result_img, face_count, det_time = detect_faces_cnn(
                    selected_image, cnn_model, confidence_threshold
                )
                st.image(result_img, caption=f"CNN - Tìm thấy {face_count} khuôn mặt", use_column_width=True)
                
                col2a, col2b, col2c = st.columns(3)
                with col2a:
                    st.metric("Số mặt phát hiện", face_count)
                with col2b:
                    st.metric("Thời gian", f"{det_time:.3f}s")
                with col2c:
                    st.metric("Tốc độ", "Nhanh" if det_time < 1 else "Chậm")
            elif detection_method == "So sánh cả hai":
                st.subheader("So sánh trực tiếp")
                
                col2a, col2b = st.columns(2)
                
                with col2a:
                    st.write("**Haar Cascade**")
                    haar_result, haar_faces, haar_time = detect_faces_haar(selected_image, haar_cascade)
                    st.image(haar_result, use_column_width=True)
                    st.write(f"**Số mặt phát hiện:** {haar_faces}")
                    st.write(f"**Thời gian:** {haar_time:.3f}s")
                
                with col2b:
                    st.write("**CNN**")
                    if cnn_model:
                        cnn_result, cnn_faces, cnn_time = detect_faces_cnn(
                            selected_image, cnn_model, confidence_threshold
                        )
                    
                    st.image(cnn_result, use_column_width=True)
                    st.write(f"**Số mặt phát hiện:** {cnn_faces}")
                    st.write(f"**Thời gian:** {cnn_time:.3f}s")
                
                # Comparison table
                st.markdown("#### Kết quả so sánh:")
                comparison_data = {
                    "Phương pháp": ["Haar Cascade", "CNN"],
                    "Số mặt phát hiện": [haar_faces, cnn_faces],
                    "Thời gian  (s)": [f"{haar_time:.3f}", f"{cnn_time:.3f}"],
                    "Tốc độ": ["Nhanh" if haar_time < cnn_time else "Chậm", 
                             "Nhanh" if cnn_time < haar_time else "Chậm"],
                }
                
                st.table(comparison_data)
        else:
            st.info("Hãy chọn một ảnh để bắt đầu detection")

if __name__ == "__main__":
    main()