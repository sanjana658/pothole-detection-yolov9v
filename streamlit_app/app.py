import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
YOLOV9_DIR = PROJECT_DIR / "yolov9"

CFG_PATH = (
    YOLOV9_DIR
    / "models"
    / "detect"
    / "yolov9-c-mwpd-enhanced.yaml"
)

WEIGHTS_PATH = (
    APP_DIR
    / "Model2_best.pt"
)

# YOLOv9 imports
sys.path.insert(
    0,
    str(YOLOV9_DIR)
)

from models.yolo import Model
from utils.augmentations import letterbox
from utils.general import non_max_suppression


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-Weather Pothole Detection",
    page_icon="🚧",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <h1 style="margin-bottom:0;">
        🚧 Multi-Weather Pothole Detection
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="font-size:1.1rem;color:#666;">
    Enhanced YOLOv9 + Grad-CAM++ Explainable AI +
    Visible Severity & Risk Assessment
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not CFG_PATH.exists():

        raise FileNotFoundError(
            f"Enhanced YOLOv9 configuration not found:\n"
            f"{CFG_PATH}"
        )

    if not WEIGHTS_PATH.exists():

        raise FileNotFoundError(
            f"Model2_best.pt not found:\n"
            f"{WEIGHTS_PATH}"
        )

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location="cpu",
        weights_only=False
    )

    model = Model(
        str(CFG_PATH),
        ch=3,
        nc=1
    )

    model.load_state_dict(
        checkpoint["model"].float().state_dict(),
        strict=False
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# INITIALIZE MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error(
        "❌ Failed to load Enhanced YOLOv9."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Project Information")

    st.write(
        "**Model:** Enhanced YOLOv9-c"
    )

    st.write(
        "**Dataset:** MWPD"
    )

    st.write(
        "**Class:** Potholes"
    )

    st.write(
        f"**Device:** `{DEVICE}`"
    )

    st.write(
        "**XAI:** Grad-CAM++"
    )

    st.divider()

    st.write(
        "### Pipeline"
    )

    st.write(
        "Image → YOLOv9 → Grad-CAM++ → "
        "Severity/Risk"
    )

    st.divider()

    st.caption(
        "The severity and risk section is a "
        "visible-image estimate. Actual pothole "
        "depth cannot be determined reliably from "
        "a single RGB image."
    )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image_rgb
):

    """
    YOLOv9-compatible letterbox preprocessing.
    """

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    resized = letterbox(
        image_bgr,
        new_shape=(640, 640),
        stride=32,
        auto=False
    )[0]

    resized_rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    tensor = (
        torch.from_numpy(
            resized_rgb
        )
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    tensor = tensor.unsqueeze(
        0
    ).to(DEVICE)

    return (
        tensor,
        resized_rgb
    )


# ============================================================
# DETECTION
# ============================================================

def detect_potholes(
    tensor,
    original_shape,
    conf_threshold=0.25,
    iou_threshold=0.45
):

    """
    Run Enhanced YOLOv9 detection.

    Output coordinates are converted back to
    the original image dimensions.

    Output format:

        x1, y1, x2, y2, confidence, class
    """

    original_h = int(
        original_shape[0]
    )

    original_w = int(
        original_shape[1]
    )


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            tensor
        )


    # --------------------------------------------------------
    # YOLOv9 DualDDetect
    #
    # For normal inference the second branch is used.
    # --------------------------------------------------------

    try:

        pred = prediction[0][1]

    except Exception:

        pred = prediction[0][0]


    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    try:

        nms_result = non_max_suppression(
            pred,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=None,
            agnostic=False,
            max_det=100
        )

        detections = nms_result[0]

    except Exception:

        # Fallback for decoded predictions.

        if (
            torch.is_tensor(pred)
            and pred.ndim == 3
        ):

            detections = pred[0]

        elif (
            torch.is_tensor(pred)
            and pred.ndim == 2
        ):

            detections = pred

        else:

            detections = torch.empty(
                (0, 6),
                device=DEVICE
            )


    if detections is None:

        return torch.empty(
            (0, 6),
            device=DEVICE
        )


    if len(detections) == 0:

        return detections


    # --------------------------------------------------------
    # Letterbox geometry
    # --------------------------------------------------------

    scale = min(
        640 / original_w,
        640 / original_h
    )

    new_w = int(
        round(original_w * scale)
    )

    new_h = int(
        round(original_h * scale)
    )

    pad_x = (
        640 - new_w
    ) // 2

    pad_y = (
        640 - new_h
    ) // 2


    # --------------------------------------------------------
    # Clone detections
    # --------------------------------------------------------

    detections = detections.clone()


    # --------------------------------------------------------
    # Convert coordinates to original image
    # --------------------------------------------------------

    detections[:, 0] = (
        detections[:, 0] - pad_x
    ) / scale

    detections[:, 1] = (
        detections[:, 1] - pad_y
    ) / scale

    detections[:, 2] = (
        detections[:, 2] - pad_x
    ) / scale

    detections[:, 3] = (
        detections[:, 3] - pad_y
    ) / scale


    # --------------------------------------------------------
    # Clip
    # --------------------------------------------------------

    detections[:, 0].clamp_(
        0,
        original_w - 1
    )

    detections[:, 1].clamp_(
        0,
        original_h - 1
    )

    detections[:, 2].clamp_(
        0,
        original_w - 1
    )

    detections[:, 3].clamp_(
        0,
        original_h - 1
    )


    return detections


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image_rgb,
    detections
):

    output = image_rgb.copy()

    if (
        detections is None
        or len(detections) == 0
    ):

        return output


    for detection in detections:

        x1, y1, x2, y2 = [
            int(v)
            for v in detection[:4]
        ]

        confidence = float(
            detection[4]
        )


        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            4
        )


        label = (
            f"Pothole "
            f"{confidence * 100:.2f}%"
        )


        cv2.putText(
            output,
            label,
            (
                x1,
                max(
                    35,
                    y1 - 12
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


    return output


# ============================================================
# GRAD-CAM++
# ============================================================

def generate_gradcam(
    tensor,
    detections,
    original_shape
):

    """
    Grad-CAM++ using model.model[37].

    This is the same working approach tested
    separately before connecting to Streamlit.

    No bounding-box masking.
    No artificial thresholding.
    """

    if (
        detections is None
        or len(detections) == 0
    ):

        return (
            None,
            "No detection available for Grad-CAM++."
        )


    target_layer = model.model[37]

    activation = None
    gradient = None


    # --------------------------------------------------------
    # Gradient hook
    # --------------------------------------------------------

    def save_gradient(
        grad
    ):

        nonlocal gradient

        gradient = grad


    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def forward_hook(
        module,
        inputs,
        output
    ):

        nonlocal activation

        if torch.is_tensor(output):

            if output.ndim == 4:

                activation = output

                output.register_hook(
                    save_gradient
                )


    handle = target_layer.register_forward_hook(
        forward_hook
    )


    try:

        model.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        with torch.enable_grad():

            prediction = model(
                tensor
            )


            # First prediction branch is
            # connected to model.model[37].

            pred = prediction[0][0]


            confidence_scores = pred[
                0,
                4,
                :
            ]


            target_index = torch.argmax(
                confidence_scores
            )


            target_score = confidence_scores[
                target_index
            ]


            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            target_score.backward()


        # ----------------------------------------------------
        # Verify activation
        # ----------------------------------------------------

        if activation is None:

            return (
                None,
                "Grad-CAM++ activation was not captured."
            )


        # ----------------------------------------------------
        # Verify gradient
        # ----------------------------------------------------

        if gradient is None:

            return (
                None,
                "Grad-CAM++ gradient was not captured."
            )


        # ----------------------------------------------------
        # Grad-CAM++ calculation
        # ----------------------------------------------------

        A = activation.detach()[0]

        G = gradient.detach()[0]

        eps = 1e-8


        # First-order gradient

        G1 = G


        # Second-order gradient

        G2 = G1 ** 2


        # Third-order gradient

        G3 = G1 ** 3


        # Activation-weighted third-order gradients

        sum_A_G3 = torch.sum(
            A * G3,
            dim=(1, 2),
            keepdim=True
        )


        # Alpha denominator

        denominator = (
            2.0 * G2
            + sum_A_G3
        )


        # Alpha coefficients

        alpha = (
            G2
            / (
                denominator
                + eps
            )
        )


        # Positive gradients

        positive_G = torch.relu(
            G1
        )


        # Channel weights

        weights = torch.sum(
            alpha * positive_G,
            dim=(1, 2)
        )


        # Weighted feature map

        cam = torch.sum(
            weights[:, None, None] * A,
            dim=0
        )


        # Positive contribution

        cam = torch.relu(
            cam
        )


        cam = (
            cam
            .detach()
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()


        # ----------------------------------------------------
        # Resize feature map to 640x640
        # ----------------------------------------------------

        cam_640 = cv2.resize(
            cam,
            (640, 640),
            interpolation=cv2.INTER_CUBIC
        )


        # ----------------------------------------------------
        # Original image geometry
        # ----------------------------------------------------

        original_h = int(
            original_shape[0]
        )

        original_w = int(
            original_shape[1]
        )


        scale = min(
            640 / original_w,
            640 / original_h
        )


        new_w = int(
            round(original_w * scale)
        )

        new_h = int(
            round(original_h * scale)
        )


        pad_x = (
            640 - new_w
        ) // 2

        pad_y = (
            640 - new_h
        ) // 2


        # ----------------------------------------------------
        # Remove letterbox padding
        # ----------------------------------------------------

        cam_content = cam_640[
            pad_y : pad_y + new_h,
            pad_x : pad_x + new_w
        ]


        # ----------------------------------------------------
        # Map back to original image
        # ----------------------------------------------------

        cam_original = cv2.resize(
            cam_content,
            (
                original_w,
                original_h
            ),
            interpolation=cv2.INTER_CUBIC
        )


        # ----------------------------------------------------
        # Smooth
        # ----------------------------------------------------

        cam_original = cv2.GaussianBlur(
            cam_original.astype(
                np.float32
            ),
            (0, 0),
            sigmaX=6
        )


        # ----------------------------------------------------
        # Normalize again
        # ----------------------------------------------------

        cam_original -= cam_original.min()

        if cam_original.max() > 0:

            cam_original /= cam_original.max()


        # ----------------------------------------------------
        # Gentle gamma
        # ----------------------------------------------------

        cam_original = np.power(
            cam_original,
            0.85
        )


        # ----------------------------------------------------
        # Convert to uint8
        # ----------------------------------------------------

        cam_uint8 = (
            cam_original * 255
        ).astype(
            np.uint8
        )


        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

        best_idx = torch.argmax(
            detections[:, 4]
        )

        detection_confidence = float(
            detections[
                best_idx,
                4
            ]
        )


        info = (
            f"Method: Grad-CAM++\n"
            f"Target layer: model.model[37]\n"
            f"Activation: "
            f"{tuple(activation.shape)}\n"
            f"Gradient: "
            f"{tuple(gradient.shape)}\n"
            f"Target index: "
            f"{int(target_index.item())}\n"
            f"Target confidence: "
            f"{float(target_score.detach()):.4f}\n"
            f"Detection confidence: "
            f"{detection_confidence:.4f}\n"
            f"Image mapping: "
            f"letterbox → original image"
        )


        return (
            cam_uint8,
            info
        )


    except Exception as e:

        return (
            None,
            "Grad-CAM++ error:\n"
            + str(e)
        )


    finally:

        handle.remove()

        model.zero_grad(
            set_to_none=True
        )


# ============================================================
# GRAD-CAM++ OVERLAY
# ============================================================

def create_gradcam_overlay(
    image_rgb,
    cam,
    detections
):

    original_h = image_rgb.shape[0]

    original_w = image_rgb.shape[1]


    # --------------------------------------------------------
    # Make sure dimensions match
    # --------------------------------------------------------

    if (
        cam.shape[0] != original_h
        or cam.shape[1] != original_w
    ):

        cam_original = cv2.resize(
            cam,
            (
                original_w,
                original_h
            ),
            interpolation=cv2.INTER_CUBIC
        )

    else:

        cam_original = cam.copy()


    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    cam_original = np.clip(
        cam_original,
        0,
        255
    ).astype(
        np.uint8
    )


    heatmap = cv2.applyColorMap(
        cam_original,
        cv2.COLORMAP_JET
    )


    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        image_rgb,
        0.62,
        heatmap,
        0.38,
        0
    )


    # --------------------------------------------------------
    # Detection box
    # --------------------------------------------------------

    if (
        detections is not None
        and len(detections) > 0
    ):

        best_idx = torch.argmax(
            detections[:, 4]
        )

        best = (
            detections[
                best_idx
            ]
            .detach()
            .cpu()
        )


        x1, y1, x2, y2 = [
            int(v)
            for v in best[:4]
        ]


        # Clip

        x1 = max(
            0,
            min(
                original_w - 1,
                x1
            )
        )

        y1 = max(
            0,
            min(
                original_h - 1,
                y1
            )
        )

        x2 = max(
            0,
            min(
                original_w - 1,
                x2
            )
        )

        y2 = max(
            0,
            min(
                original_h - 1,
                y2
            )
        )


        confidence = float(
            best[4]
        )


        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            4
        )


        label = (
            f"Pothole "
            f"{confidence * 100:.2f}%"
        )


        cv2.putText(
            overlay,
            label,
            (
                x1,
                max(
                    35,
                    y1 - 12
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


    return overlay


# ============================================================
# DYNAMIC VISIBLE SEVERITY / RISK
# ============================================================

def run_vlm_analysis(
    image_pil,
    confidence,
    detections
):

    """
    Dynamic visible-severity assessment.

    IMPORTANT:
    This is an image-based heuristic assessment.
    It does not determine actual physical pothole depth.
    """

    image_rgb = np.array(
        image_pil.convert("RGB")
    )

    image_h, image_w = (
        image_rgb.shape[:2]
    )


    # --------------------------------------------------------
    # Best detection
    # --------------------------------------------------------

    best_idx = torch.argmax(
        detections[:, 4]
    )

    best = (
        detections[
            best_idx
        ]
        .detach()
        .cpu()
        .numpy()
    )


    x1, y1, x2, y2 = [
        int(v)
        for v in best[:4]
    ]


    # Clip

    x1 = max(
        0,
        min(
            image_w - 1,
            x1
        )
    )

    y1 = max(
        0,
        min(
            image_h - 1,
            y1
        )
    )

    x2 = max(
        0,
        min(
            image_w - 1,
            x2
        )
    )

    y2 = max(
        0,
        min(
            image_h - 1,
            y2
        )
    )


    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    box_width = max(
        1,
        x2 - x1
    )

    box_height = max(
        1,
        y2 - y1
    )


    box_area = (
        box_width
        * box_height
    )

    image_area = (
        image_w
        * image_h
    )


    area_ratio = (
        box_area
        / image_area
    )


    width_ratio = (
        box_width
        / image_w
    )


    # --------------------------------------------------------
    # Crop detected region
    # --------------------------------------------------------

    crop = image_rgb[
        y1:y2,
        x1:x2
    ]


    if crop.size == 0:

        return """
### Visual Severity Assessment

Unable to analyze the detected region.
"""


    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    crop_bgr = cv2.cvtColor(
        crop,
        cv2.COLOR_RGB2BGR
    )

    hsv = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2GRAY
    )


    brightness = hsv[:, :, 2]

    saturation = hsv[:, :, 1]


    # --------------------------------------------------------
    # Dark region
    # --------------------------------------------------------

    dark_ratio = float(
        np.mean(
            brightness < 80
        )
    )


    # --------------------------------------------------------
    # Edge ratio
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_ratio = float(
        np.mean(
            edges > 0
        )
    )


    # --------------------------------------------------------
    # Water-like visual region
    #
    # This is only a heuristic.
    # --------------------------------------------------------

    water_like = (
        (brightness < 110)
        &
        (saturation < 100)
    )


    water_like_ratio = float(
        np.mean(
            water_like
        )
    )


    # ========================================================
    # SCORE
    # ========================================================

    score = 0.0

    reasons = []


    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    if area_ratio >= 0.20:

        score += 3

        reasons.append(
            "The detected pothole occupies a large "
            "portion of the image."
        )

    elif area_ratio >= 0.08:

        score += 2

        reasons.append(
            "The detected pothole has a moderate-to-large "
            "visible area."
        )

    elif area_ratio >= 0.03:

        score += 1

        reasons.append(
            "The detected pothole is relatively small-to-moderate."
        )

    else:

        score += 0.5

        reasons.append(
            "The detected pothole occupies a relatively small "
            "image area."
        )


    # --------------------------------------------------------
    # Width
    # --------------------------------------------------------

    if width_ratio >= 0.40:

        score += 2

        reasons.append(
            "The pothole spans a substantial portion of "
            "the road view."
        )

    elif width_ratio >= 0.20:

        score += 1


    # --------------------------------------------------------
    # Dark/wet appearance
    # --------------------------------------------------------

    if dark_ratio >= 0.45:

        score += 1.5

        reasons.append(
            "A large dark region is visible inside "
            "the detected pothole."
        )

    elif dark_ratio >= 0.25:

        score += 0.75


    # --------------------------------------------------------
    # Irregular/broken edges
    # --------------------------------------------------------

    if edge_ratio >= 0.15:

        score += 1.5

        reasons.append(
            "Strong irregular edges or broken pavement "
            "are visible."
        )

    elif edge_ratio >= 0.08:

        score += 0.75


    # --------------------------------------------------------
    # Water-like appearance
    # --------------------------------------------------------

    if water_like_ratio >= 0.45:

        score += 1.5

        reasons.append(
            "The detected region has a strong dark, "
            "low-saturation appearance consistent with "
            "visible water or a wet surface."
        )

    elif water_like_ratio >= 0.30:

        score += 0.75


    # ========================================================
    # SEVERITY
    # ========================================================

    if score >= 6.0:

        severity = "HIGH"
        severity_icon = "🔴"

    elif score >= 3.5:

        severity = "MODERATE"
        severity_icon = "🟠"

    else:

        severity = "LOW"
        severity_icon = "🟢"


    # ========================================================
    # RISK
    # ========================================================

    if score >= 6.0:

        risk = "HIGH"
        risk_icon = "🔴"

    elif score >= 3.5:

        risk = "MEDIUM"
        risk_icon = "🟠"

    else:

        risk = "LOW"
        risk_icon = "🟢"


    # --------------------------------------------------------
    # Confidence description
    # --------------------------------------------------------

    if confidence >= 0.85:

        confidence_text = (
            "The YOLOv9 detection has high confidence."
        )

    elif confidence >= 0.60:

        confidence_text = (
            "The YOLOv9 detection has moderate confidence."
        )

    else:

        confidence_text = (
            "The YOLOv9 detection has relatively low confidence."
        )


    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    reason_text = "\n".join(
        f"- {reason}"
        for reason in reasons
    )


    # ========================================================
    # REPORT
    # ========================================================

    return f"""
### 🔍 Visual Severity & Risk Assessment

**Model:** Enhanced YOLOv9-c

**Detection confidence:** {confidence * 100:.2f}%

---

### 1. Detection

**Pothole detected:** Yes

{confidence_text}

---

### 2. Visible Characteristics

- Bounding-box width: **{box_width}px**
- Bounding-box height: **{box_height}px**
- Approximate image coverage: **{area_ratio * 100:.1f}%**
- Dark-region ratio: **{dark_ratio * 100:.1f}%**
- Edge ratio: **{edge_ratio * 100:.1f}%**
- Water-like visual ratio: **{water_like_ratio * 100:.1f}%**

---

### 3. Estimated Visible Severity

## {severity_icon} {severity}

{reason_text}

---

### 4. Estimated Safety Risk

## {risk_icon} {risk}

The risk estimate is based on visible characteristics
of the detected road defect, including its apparent
size, appearance and surrounding visual damage.

---

### 5. Visual Reasoning

The assessment considers:

- pothole size relative to the image
- visible width and height
- dark/wet appearance
- irregular pavement edges
- visible road-surface damage
- YOLOv9 detection confidence

> **Important:** This is a visible severity/risk estimate
> from a single RGB image. Actual pothole depth, structural
> damage and physical dimensions cannot be reliably determined
> from the image alone.
"""


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a road image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# ANALYSIS FORM
# ============================================================

with st.form(
    "analysis_form"
):

    analyze_button = st.form_submit_button(
        "🚀 Analyze Pothole",
        type="primary"
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

if analyze_button:

    if uploaded_file is None:

        st.warning(
            "Please upload a road image first."
        )

        st.stop()


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image_pil = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )

    image_rgb = np.array(
        image_pil
    )


    # ========================================================
    # INPUT
    # ========================================================

    st.subheader(
        "📷 Input Image"
    )

    st.image(
        image_rgb,
        use_container_width=True
    )


    # ========================================================
    # PREPROCESS
    # ========================================================

    tensor, model_image = preprocess_image(
        image_rgb
    )


    # ========================================================
    # YOLOv9 DETECTION
    # ========================================================

    with st.spinner(
        "🔍 Running Enhanced YOLOv9..."
    ):

        detections = detect_potholes(
            tensor,
            image_rgb.shape
        )


    # ========================================================
    # NO DETECTION
    # ========================================================

    if (
        detections is None
        or len(detections) == 0
    ):

        st.error(
            "❌ No pothole detected."
        )

        st.info(
            "Try another road image or an image "
            "where the pothole is more visible."
        )

        st.stop()


    # ========================================================
    # BEST DETECTION
    # ========================================================

    best_idx = torch.argmax(
        detections[:, 4]
    )

    best_detection = (
        detections[
            best_idx
        ]
        .detach()
        .cpu()
    )

    confidence = float(
        best_detection[4]
    )


    # ========================================================
    # DETECTION RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Enhanced YOLOv9 Detection"
    )


    detection_image = draw_detections(
        image_rgb,
        detections
    )


    col1, col2 = st.columns(
        [2.2, 1]
    )


    with col1:

        st.image(
            detection_image,
            caption="Pothole Detection",
            use_container_width=True
        )


    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.metric(
            "Potholes Detected",
            len(detections)
        )


        if confidence >= 0.85:

            st.success(
                "High-confidence detection"
            )

        elif confidence >= 0.60:

            st.warning(
                "Moderate-confidence detection"
            )

        else:

            st.warning(
                "Low-confidence detection"
            )


    # ========================================================
    # GRAD-CAM++
    # ========================================================

    st.divider()

    st.subheader(
        "🔥 Grad-CAM++ Explainable AI"
    )


    with st.spinner(
        "Generating Grad-CAM++ explanation..."
    ):

        cam, cam_info = generate_gradcam(
            tensor,
            detections,
            image_rgb.shape
        )


    if cam is not None:

        gradcam_image = create_gradcam_overlay(
            image_rgb,
            cam,
            detections
        )


        st.image(
            gradcam_image,
            caption=(
                "Grad-CAM++ — regions contributing "
                "to the Enhanced YOLOv9 prediction"
            ),
            use_container_width=True
        )


        st.success(
            "✅ Grad-CAM++ generated successfully."
        )


        with st.expander(
            "🔍 Grad-CAM++ Technical Information"
        ):

            st.code(
                cam_info
            )


        st.info(
            "Grad-CAM++ visualizes regions contributing "
            "to the model's prediction. Stronger activation "
            "indicates regions associated with the prediction; "
            "it should not be interpreted as causal proof."
        )


    else:

        st.warning(
            cam_info
        )


    # ========================================================
    # VISUAL SEVERITY / RISK
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 Visual Severity & Risk Assessment"
    )


    with st.spinner(
        "Analyzing visible pothole characteristics..."
    ):

        report = run_vlm_analysis(
            image_pil,
            confidence,
            detections
        )


    st.markdown(
        report
    )


    # ========================================================
    # FINAL PIPELINE
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Analysis Pipeline"
    )


    st.markdown(
        """
        **Road Image**
        ↓
        **Enhanced YOLOv9**
        ↓
        **Pothole Detection**
        ↓
        **Grad-CAM++ Explainability**
        ↓
        **Visible Severity Assessment**
        ↓
        **Safety-Risk Estimation**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Multi-Weather Pothole Detection using Enhanced YOLOv9 "
    "and Grad-CAM++"
)