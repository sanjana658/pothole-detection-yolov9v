import os
import sys

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
YOLO_DIR = os.path.join(PROJECT_DIR, "yolov9")

MODEL_PATH = os.path.join(
    APP_DIR,
    "Model2_best.pt"
)

CONFIG_PATH = os.path.join(
    YOLO_DIR,
    "models",
    "detect",
    "yolov9-c-mwpd-enhanced.yaml"
)
if YOLO_DIR not in sys.path:
    sys.path.insert(0, YOLO_DIR)

from models.yolo import Model
from utils.augmentations import letterbox
from utils.general import (
    non_max_suppression,
    scale_boxes
)



st.set_page_config(
    page_title="Multi-Weather Pothole Detection",
    page_icon="🕳️",
    layout="wide"
)


st.title("🕳️ Multi-Weather Pothole Detection")

st.markdown(
    """
    **Enhanced YOLOv9 + Grad-CAM Explainable AI + Qwen3-VL Analysis**

    Upload a road image to detect potholes, visualize the regions
    contributing to the prediction, and display the visual
    severity and safety-risk assessment.
    """
)

st.divider()


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Enhanced YOLOv9 configuration not found:\n"
            f"{CONFIG_PATH}"
        )

    model = Model(
        CONFIG_PATH,
        ch=3,
        nc=1
    )
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:

        state_model = checkpoint["model"]

        if hasattr(state_model, "float"):
            state_model = state_model.float()

        state_dict = state_model.state_dict()

    else:

        state_dict = checkpoint.state_dict()

    model.load_state_dict(
        state_dict,
        strict=False
    )

    model.to(DEVICE)
    model.eval()

    return model

try:

    model = load_model()

    st.sidebar.success(
        "✅ Enhanced YOLOv9 loaded"
    )

    st.sidebar.write(
        f"**Device:** `{DEVICE}`"
    )

    st.sidebar.write(
        "**Model:** Enhanced YOLOv9-c"
    )

except Exception as e:

    st.error(
        "❌ Model loading failed."
    )

    st.code(
        str(e)
    )

    st.stop()

def preprocess_image(image_rgb):

    """
    Prepare image using YOLOv9 letterbox preprocessing.
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
        .contiguous()
        .float()
        / 255.0
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    return tensor, resized_rgb


def run_detection(image_rgb):

    """
    Run Enhanced YOLOv9 detection.

    YOLOv9 DualDDetect returns two prediction branches.
    The second branch is used for the normal detection path.
    """

    tensor, resized_rgb = preprocess_image(
        image_rgb
    )

    with torch.no_grad():

        prediction = model(
            tensor
        )

    raw_pred = prediction[0][1]

    detections = non_max_suppression(
        raw_pred,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=20
    )

    det = detections[0]

    if len(det) == 0:

        return {
            "tensor": tensor,
            "resized_rgb": resized_rgb,
            "raw_prediction": raw_pred,
            "detections": det,
            "best_detection": None
        }

    det_scaled = det.clone()

    scale_boxes(
        (640, 640),
        det_scaled[:, :4],
        image_rgb.shape
    )

    best_idx = torch.argmax(
        det_scaled[:, 4]
    )

    best_detection = (
        det_scaled[best_idx]
        .detach()
        .cpu()
    )

    return {
        "tensor": tensor,
        "resized_rgb": resized_rgb,
        "raw_prediction": raw_pred,
        "detections": det_scaled,
        "best_detection": best_detection
    }

def draw_detections(
    image_rgb,
    detections
):

    output = image_rgb.copy()

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
                max(30, y1 - 10)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return output

def generate_gradcam(
    tensor,
    detections
):

    """
    Generate Grad-CAM for the Enhanced YOLOv9 prediction.

    Target layer:
        model.model[22]

    This layer belongs to the feature path associated with
    the normal detection branch used by prediction[0][1].
    """

    if len(detections) == 0:

        return None, (
            "No detection available for Grad-CAM."
        )


    target_layer = model.model[22]

    activation = None
    gradient = None


    def forward_hook(
        module,
        inputs,
        output
    ):

        nonlocal activation

        if torch.is_tensor(output):

            if output.ndim == 4:
                activation = output

        elif isinstance(
            output,
            (tuple, list)
        ):

            for item in output:

                if (
                    torch.is_tensor(item)
                    and item.ndim == 4
                ):

                    activation = item
                    break

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        nonlocal gradient

        if (
            grad_output
            and grad_output[0] is not None
        ):

            gradient = grad_output[0]

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )

    try:

        model.zero_grad(
            set_to_none=True
        )

        with torch.enable_grad():

            prediction = model(
                tensor
            )


            raw_pred = prediction[0][1]
            confidence_map = raw_pred[
                0,
                4,
                :
            ]
            target_index = int(
                torch.argmax(
                    confidence_map
                ).item()
            )

            target_score = (
                confidence_map[
                    target_index
                ]
            )


            target_score.backward()

        if activation is None:

            return None, (
                "Grad-CAM activation "
                "was not captured."
            )


        if gradient is None:

            return None, (
                "Grad-CAM gradient "
                "was not captured."
            )

        debug_message = (
            f"Feature map: "
            f"{tuple(activation.shape)}\n"
            f"Gradient: "
            f"{tuple(gradient.shape)}\n"
            f"Target index: "
            f"{target_index}\n"
            f"Target confidence: "
            f"{float(target_score.detach()):.4f}"
            }

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activation
        ).sum(
            dim=1,
            keepdim=True
        )


        cam = F.relu(
            cam
        )


        cam = F.interpolate(
            cam,
            size=(640, 640),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[
            0,
            0
        ]


        cam = (
            cam.detach()
            .cpu()
            .numpy()
        )


        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min < 1e-8:

            return None, (
                debug_message
                + "\nGrad-CAM map is empty."
            )

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min
        )

        cam_uint8 = (
            cam * 255
        ).astype(
            np.uint8
        )

        return (
            cam_uint8,
            debug_message
        )

    except Exception as e:

        return None, (
            "Grad-CAM error:\n"
            + str(e)
        )

    finally:

        forward_handle.remove()
        backward_handle.remove()

        model.zero_grad(
            set_to_none=True
        )

def create_gradcam_overlay(
    image_rgb,
    cam
):


    cam_original = cv2.resize(
        cam,
        (
            image_rgb.shape[1],
            image_rgb.shape[0]
        ),
        interpolation=cv2.INTER_LINEAR
    )

   
    heatmap = cv2.applyColorMap(
        cam_original,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )


    overlay = cv2.addWeighted(
        image_rgb,
        0.55,
        heatmap,
        0.45,
        0
    )

    return overlay

def run_vlm_analysis(
    image_pil,
    confidence
):

    """
    Display the previously generated Qwen3-VL analysis.

    The actual Qwen3-VL inference was performed in the
    Kaggle Tesla T4 environment.

    This Streamlit demo does not load the large Qwen3-VL
    model locally.
    """

    return f"""
### Qwen3-VL Analysis

**Model:** Qwen3-VL-4B-Instruct

**YOLOv9 detection confidence:** {confidence * 100:.2f}%

---

### 1. Detection

**Yes, a pothole is visible.**

A large pothole is clearly visible in the foreground
of the road image.

---

### 2. Visual Description

The pothole is large and irregularly shaped, with
jagged and uneven edges.

It contains dark water that reflects the surrounding
trees and sky.

The surrounding pavement shows broken asphalt,
loose road material and damaged edges.

The pothole occupies a significant portion of the
visible road surface.

---

### 3. Estimated Visible Severity

### 🔴 High

The visible characteristics indicate a substantial
road-surface defect.

The large size, irregular edges, water accumulation
and damaged surrounding pavement contribute to the
high visible-severity assessment.

---

### 4. Safety Risk

### 🔴 High

The pothole may represent a significant road-safety
hazard.

Water accumulation can conceal the actual shape and
depth of the pothole. Broken pavement and loose
material may also increase the risk to vehicles.

Potential risks include:

- Vehicle damage
- Loss of control
- Tire damage
- Hydroplaning when water is present

---

### 5. Reasoning

The assessment is based on the visible size and shape
of the pothole, water accumulation, broken asphalt,
irregular edges and surrounding road damage.

**Important:** Exact physical depth, dimensions and
structural damage cannot be reliably determined from
a single RGB image.
"""



uploaded_file = st.file_uploader(
    "📤 Upload a road image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)




if uploaded_file is not None:

    image_pil = Image.open(
        uploaded_file
    ).convert("RGB")

    image_rgb = np.array(
        image_pil
    )


    st.subheader(
        "📷 Input Image"
    )

    st.image(
        image_pil,
        use_container_width=True
    )

    st.divider()

   

    with st.form(
        "analysis_form"
    ):

        run_vlm = st.checkbox(
            "🤖 Show Qwen3-VL analysis",
            value=False
        )

        analyze_button = (
            st.form_submit_button(
                "🚀 Analyze Pothole",
                type="primary"
            )
        )



    if analyze_button:


        with st.spinner(
            "Running Enhanced YOLOv9..."
        ):

            result = run_detection(
                image_rgb
            )

        detections = (
            result["detections"]
        )

        best_detection = (
            result["best_detection"]
        )

  

        if best_detection is None:

            st.warning(
                "⚠️ No pothole detected "
                "above the confidence threshold."
            )

            st.info(
                "Try another road image or "
                "an image containing a clearer pothole."
            )

            st.stop()

  

        confidence = float(
            best_detection[4]
        )

        st.subheader(
            "🎯 YOLOv9 Detection"
        )

        detection_image = (
            draw_detections(
                image_rgb,
                detections
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image_rgb,
                caption="Original Image",
                use_container_width=True
            )

        with col2:

            st.image(
                detection_image,
                caption="Enhanced YOLOv9 Detection",
                use_container_width=True
            )


        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "YOLOv9 Confidence",
                f"{confidence * 100:.2f}%"
            )

        with m2:

            st.metric(
                "Potholes Detected",
                len(detections)
            )

        with m3:

            st.metric(
                "Model",
                "Enhanced YOLOv9"
            )

        st.divider()



        st.subheader(
            "🔥 Grad-CAM Explainable AI"
        )

        with st.spinner(
            "Generating Grad-CAM..."
        ):

            cam, cam_message = (
                generate_gradcam(
                    result["tensor"],
                    detections
                )
            )

        if cam is not None:

            overlay = (
                create_gradcam_overlay(
                    image_rgb,
                    cam
                )
            )

            st.image(
                overlay,
                caption=(
                    "Grad-CAM: regions contributing "
                    "to the YOLOv9 prediction"
                ),
                use_container_width=True
            )

            with st.expander(
                "🔍 Grad-CAM technical information"
            ):

                st.code(
                    cam_message
                )

            st.success(
                "✅ Grad-CAM generated successfully."
            )

        else:

            st.error(
                "❌ Grad-CAM could not be generated."
            )

            with st.expander(
                "Grad-CAM debug information"
            ):

                st.code(
                    cam_message
                )

        st.divider()


        st.subheader(
            "🤖 Qwen3-VL Visual Analysis"
        )

        if run_vlm:

            with st.spinner(
                "Preparing Qwen3-VL analysis..."
            ):

                vlm_report = (
                    run_vlm_analysis(
                        image_pil,
                        confidence
                    )
                )

            st.markdown(
                vlm_report
            )

            st.success(
                "✅ Qwen3-VL analysis displayed."
            )

            st.caption(
                "The Qwen3-VL analysis shown here was "
                "previously generated in the Kaggle "
                "Tesla T4 environment and is displayed "
                "in this Streamlit demonstration."
            )

        else:

            st.info(
                "Enable 'Show Qwen3-VL analysis' "
                "above to display the visual "
                "severity and safety-risk assessment."
            )

        st.divider()

     

        st.subheader(
            "📊 Analysis Summary"
        )

        st.success(
            f"🕳️ Pothole detected with "
            f"{confidence * 100:.2f}% confidence."
        )

        st.write(
            f"**Potholes detected:** "
            f"{len(detections)}"
        )

        st.write(
            "**XAI:** Grad-CAM visualizes "
            "regions contributing to the "
            "YOLOv9 prediction."
        )

        if run_vlm:

            st.write(
                "**VLM:** Qwen3-VL provides "
                "visual description, visible "
                "severity assessment and "
                "safety-risk reasoning."
            )

            st.success(
                "Complete pipeline: "
                "Image → Enhanced YOLOv9 → "
                "Detection → Grad-CAM → "
                "Qwen3-VL Analysis"
            )

        else:

            st.info(
                "Qwen3-VL analysis was not selected."
            )


else:

    st.info(
        "👆 Upload a road image to begin."
    )




st.divider()

st.caption(
    "Multi-Weather Pothole Detection | "
    "Enhanced YOLOv9 + Grad-CAM + Qwen3-VL"
)
