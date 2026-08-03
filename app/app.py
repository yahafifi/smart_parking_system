import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Smart Parking YOLO",
    page_icon="🚗",
    layout="wide"
)

# =========================
# PATHS
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    return YOLO(MODEL_PATH)

model = load_model()

# =========================
# TITLE
# =========================
st.title("🚗 Smart Parking Detection using YOLO")
st.write("Detect *empty* and *occupied* parking spaces using a trained YOLO model.")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

mode = st.sidebar.radio(
    "Choose Detection Mode",
    ["Image Upload", "Live Camera"]
)

# =========================
# COUNT FUNCTION
# =========================
def count_slots(results):
    empty_count = 0
    occupied_count = 0

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name == "empty":
            empty_count += 1
        elif class_name == "occupied":
            occupied_count += 1

    return empty_count, occupied_count

# =========================
# IMAGE UPLOAD MODE
# =========================
if mode == "Image Upload":
    st.header("📷 Image Upload Detection")

    uploaded_file = st.file_uploader(
        "Upload parking image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        results = model.predict(
            source=temp_path,
            conf=conf_threshold
        )

        result_image = results[0].plot()
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        empty_count, occupied_count = count_slots(results)

        with col2:
            st.subheader("Detection Result")
            st.image(result_image, use_container_width=True)

        st.success(f"🟢 Empty Slots: {empty_count}")
        st.error(f"🔴 Occupied Slots: {occupied_count}")

# =========================
# LIVE CAMERA MODE
# =========================
elif mode == "Live Camera":
    st.header("🎥 Live Camera Detection")

    st.warning("Make sure your camera is connected and not used by another app.")

    start_button = st.button("Start Camera")
    stop_button = st.button("Stop Camera")

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    if start_button:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not open camera.")
        else:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    st.error("Failed to read frame from camera.")
                    break

                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    verbose=False
                )

                result_frame = results[0].plot()
                result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

                empty_count, occupied_count = count_slots(results)

                frame_placeholder.image(
                    result_frame,
                    channels="RGB",
                    use_container_width=True
                )

                stats_placeholder.markdown(
                    f"""
                    ### Detection Summary
                    🟢 Empty Slots: *{empty_count}*  
                    🔴 Occupied Slots: *{occupied_count}*
                    """
                )

                if stop_button:
                    break

            cap.release()