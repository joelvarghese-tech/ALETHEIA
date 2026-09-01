import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tempfile
import os


# ============================================================
# VERITAS
# Lightweight Biological-Signal Screening
# ============================================================

st.set_page_config(
    page_title="VERITAS",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# PAGE STYLING
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 0;
}

.subtitle {
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    margin-top: 20px;
}

.safe {
    background-color: #e8f5e9;
    border: 2px solid #43a047;
}

.warning {
    background-color: #ffebee;
    border: 2px solid #e53935;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh


# MediaPipe landmark numbers for the eyes
LEFT_EYE = [
    33, 160, 158, 133, 153, 144
]

RIGHT_EYE = [
    362, 385, 387, 263, 373, 380
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def distance(p1, p2):
    """
    Calculate the distance between two points.
    """

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR).

    EAR =
    (vertical distance 1 + vertical distance 2)
    ------------------------------------------------
                 2 × horizontal distance
    """

    vertical_1 = distance(
        eye_points[1],
        eye_points[5]
    )

    vertical_2 = distance(
        eye_points[2],
        eye_points[4]
    )

    horizontal = distance(
        eye_points[0],
        eye_points[3]
    )

    if horizontal == 0:
        return 0

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return ear


def get_eye_points(
    face_landmarks,
    indexes,
    width,
    height
):
    """
    Convert MediaPipe landmark coordinates
    into normal pixel coordinates.
    """

    points = []

    for index in indexes:

        landmark = face_landmarks.landmark[index]

        x = landmark.x * width
        y = landmark.y * height

        points.append(
            (x, y)
        )

    return points


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyse_video(video_path):

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        return {
            "success": False,
            "error": "Could not open the video."
        }

    # Get video information
    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # Store EAR measurements
    ear_values = []
    timestamps = []

    frame_number = 0
    face_frames = 0

    # Blink information
    blink_count = 0

    eyes_closed = False
    closed_frames = 0

    # Prototype threshold
    EAR_THRESHOLD = 0.20

    # Minimum number of closed frames
    # required before considering it a blink
    MIN_CLOSED_FRAMES = 2

    # Progress bar
    progress = st.progress(0)

    # Create MediaPipe face detector
    with mp_face_mesh.FaceMesh(

        static_image_mode=False,

        max_num_faces=1,

        refine_landmarks=True,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5

    ) as face_mesh:

        # ====================================================
        # READ EVERY VIDEO FRAME
        # ====================================================

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # OpenCV uses BGR.
            # MediaPipe expects RGB.
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Detect face landmarks
            results = face_mesh.process(rgb)

            # If a face was detected
            if results.multi_face_landmarks:

                face_frames += 1

                face = (
                    results.multi_face_landmarks[0]
                )

                height, width, _ = frame.shape

                # Get eye points
                left_eye = get_eye_points(
                    face,
                    LEFT_EYE,
                    width,
                    height
                )

                right_eye = get_eye_points(
                    face,
                    RIGHT_EYE,
                    width,
                    height
                )

                # Calculate EAR for both eyes
                left_ear = calculate_ear(
                    left_eye
                )

                right_ear = calculate_ear(
                    right_eye
                )

                # Average the two eyes
                ear = (
                    left_ear +
                    right_ear
                ) / 2

                # Store EAR
                ear_values.append(ear)

                # Store timestamp
                timestamps.append(
                    frame_number / fps
                )

                # =================================================
                # BLINK DETECTION
                # =================================================

                if ear < EAR_THRESHOLD:

                    closed_frames += 1

                else:

                    # Eye has opened again
                    if (
                        closed_frames >=
                        MIN_CLOSED_FRAMES
                    ):

                        blink_count += 1

                    closed_frames = 0
                    eyes_closed = False

                if (
                    closed_frames >=
                    MIN_CLOSED_FRAMES
                ):

                    eyes_closed = True

            # Update progress
            if (
                total_frames > 0
                and frame_number % 10 == 0
            ):

                progress.progress(
                    min(
                        frame_number /
                        total_frames,
                        1.0
                    )
                )

    progress.progress(1.0)

    cap.release()

    # =========================================================
    # CHECK WHETHER ENOUGH DATA WAS FOUND
    # =========================================================

    if len(ear_values) < 10:

        return {
            "success": False,
            "error":
                "Could not detect enough facial data. "
                "Try a clearer video with a visible face."
        }

    # Convert EAR list to NumPy array
    ear_array = np.array(
        ear_values
    )

    # Video duration
    duration = (
        timestamps[-1]
        if timestamps
        else 0
    )

    # Average EAR
    average_ear = float(
        np.mean(ear_array)
    )

    # Minimum EAR
    minimum_ear = float(
        np.min(ear_array)
    )

    # =========================================================
    # BLINK ANALYSIS
    # =========================================================

    # This is only a rough prototype estimate.
    expected_min_blinks = max(
        1,
        int(duration / 12)
    )

    if duration >= 5:

        blink_ratio = (
            blink_count /
            expected_min_blinks
        )

        if blink_ratio < 0.25:

            blink_anomaly = 0.75

        elif blink_ratio < 0.50:

            blink_anomaly = 0.40

        else:

            blink_anomaly = 0.05

    else:

        # Short videos are less reliable
        blink_anomaly = 0.20

    # =========================================================
    # EAR VARIATION
    # =========================================================

    ear_std = float(
        np.std(ear_array)
    )

    if ear_std < 0.015:

        variation_anomaly = 0.70

    elif ear_std < 0.025:

        variation_anomaly = 0.35

    else:

        variation_anomaly = 0.05

    # =========================================================
    # COMBINE ANOMALY SIGNALS
    # =========================================================

    anomaly_score = (

        blink_anomaly * 0.65

        +

        variation_anomaly * 0.35

    )

    anomaly_score = float(
        np.clip(
            anomaly_score,
            0,
            1
        )
    )

    # Convert anomaly into a
    # user-friendly prototype score
    trust_score = int(
        round(
            (1 - anomaly_score)
            * 100
        )
    )

    # =========================================================
    # CLASSIFICATION
    # =========================================================

    if anomaly_score >= 0.60:

        classification = (
            "SUSPICIOUS BIOLOGICAL PATTERN"
        )

        status = "warning"

    else:

        classification = (
            "LOW BIOLOGICAL ANOMALY"
        )

        status = "safe"

    # =========================================================
    # CREATE DATAFRAME
    # =========================================================

    data = pd.DataFrame({

        "Time": timestamps,

        "EAR": ear_values

    })

    # =========================================================
    # RETURN RESULTS
    # =========================================================

    return {

        "success": True,

        "data": data,

        "frames": frame_number,

        "face_frames": face_frames,

        "duration": duration,

        "blink_count": blink_count,

        "average_ear": average_ear,

        "minimum_ear": minimum_ear,

        "ear_std": ear_std,

        "anomaly_score": anomaly_score,

        "trust_score": trust_score,

        "classification": classification,

        "status": status
    }


# ============================================================
# USER INTERFACE
# ============================================================

st.markdown(
    '<div class="main-title">VERITAS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Lightweight Biological-Signal '
    'Screening for Suspicious Video'
    '</div>',
    unsafe_allow_html=True
)


# Important disclaimer
st.info(
    "VERITAS is a prototype screening system. "
    "Its result is an anomaly indicator, "
    "not proof that a video is authentic or fake."
)


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_file = st.file_uploader(

    "Upload a video for analysis",

    type=[
        "mp4",
        "mov",
        "avi",
        "mkv"
    ]
)


# ============================================================
# AFTER VIDEO UPLOAD
# ============================================================

if uploaded_file is not None:

    # Show uploaded video
    st.video(
        uploaded_file
    )

    st.markdown("---")

    # Analyse button
    if st.button(
        "🔍 ANALYSE VIDEO",
        use_container_width=True
    ):

        # ====================================================
        # SAVE TEMPORARY VIDEO
        # ====================================================

        suffix = os.path.splitext(
            uploaded_file.name
        )[1]

        with tempfile.NamedTemporaryFile(

            delete=False,

            suffix=suffix

        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            video_path = (
                temp_file.name
            )

        try:

            # =================================================
            # RUN ANALYSIS
            # =================================================

            with st.spinner(
                "Analysing facial biological signals..."
            ):

                result = analyse_video(
                    video_path
                )

            # =================================================
            # HANDLE ERROR
            # =================================================

            if not result["success"]:

                st.error(
                    result["error"]
                )

            else:

                st.success(
                    "Analysis complete."
                )

                # =============================================
                # METRICS
                # =============================================

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                with col1:

                    st.metric(
                        "Blink Events",
                        result["blink_count"]
                    )

                with col2:

                    st.metric(
                        "Frames Analysed",
                        result["frames"]
                    )

                with col3:

                    st.metric(
                        "Average EAR",
                        f"{result['average_ear']:.3f}"
                    )

                with col4:

                    st.metric(
                        "Trust Score",
                        f"{result['trust_score']}%"
                    )

                # =============================================
                # EAR GRAPH
                # =============================================

                st.subheader(
                    "👁 Eye Aspect Ratio Over Time"
                )

                df = result["data"]

                fig = go.Figure()

                fig.add_trace(

                    go.Scatter(

                        x=df["Time"],

                        y=df["EAR"],

                        mode="lines",

                        name="EAR"

                    )

                )

                # Blink threshold
                fig.add_hline(

                    y=0.20,

                    line_dash="dash",

                    annotation_text=
                        "Blink threshold"

                )

                fig.update_layout(

                    xaxis_title=
                        "Time (seconds)",

                    yaxis_title=
                        "Eye Aspect Ratio",

                    height=450

                )

                st.plotly_chart(

                    fig,

                    use_container_width=True

                )

                # =============================================
                # RESULT
                # =============================================

                if result["status"] == "safe":

                    st.markdown(

                        f"""
                        <div class="result-box safe">

                            <h2>
                            🟢 LOW BIOLOGICAL ANOMALY
                            </h2>

                            <p>
                            VERITAS detected eye-behaviour
                            patterns with relatively normal
                            EAR variation.
                            </p>

                            <h3>
                            Trust Score:
                            {result['trust_score']}%
                            </h3>

                        </div>
                        """,

                        unsafe_allow_html=True

                    )

                else:

                    st.markdown(

                        f"""
                        <div class="result-box warning">

                            <h2>
                            🔴 SUSPICIOUS BIOLOGICAL PATTERN
                            </h2>

                            <p>
                            VERITAS detected an unusually low
                            or inconsistent eye-behaviour signal.
                            </p>

                            <h3>
                            Trust Score:
                            {result['trust_score']}%
                            </h3>

                        </div>
                        """,

                        unsafe_allow_html=True

                    )

                # =============================================
                # TECHNICAL DETAILS
                # =============================================

                with st.expander(
                    "View technical analysis"
                ):

                    st.write(
                        f"Video duration: "
                        f"{result['duration']:.2f} seconds"
                    )

                    st.write(
                        f"Frames with detected face: "
                        f"{result['face_frames']}"
                    )

                    st.write(
                        f"Minimum EAR: "
                        f"{result['minimum_ear']:.3f}"
                    )

                    st.write(
                        f"EAR variation: "
                        f"{result['ear_std']:.4f}"
                    )

                    st.write(
                        f"Biological anomaly score: "
                        f"{result['anomaly_score']:.2f}"
                    )

                    st.caption(

                        "The current scoring system is a "
                        "prototype heuristic and has not "
                        "been scientifically validated as "
                        "a production deepfake classifier."

                    )

        finally:

            # Delete temporary video
            if os.path.exists(
                video_path
            ):

                os.remove(
                    video_path
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "VERITAS v1.0 • Computer Vision Prototype • "
    "Eye Aspect Ratio / Blink Analysis"
)
