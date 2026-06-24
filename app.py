import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet



st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
    135deg,
    #0b3d2e,
    #14532d,
    #1b4332);
}

section[data-testid="stSidebar"]{
    background: rgba(0,0,0,0.25);
    backdrop-filter: blur(12px);
}

.hero{
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(15px);
    border:1px solid rgba(255,255,255,0.15);
    border-radius:25px;
    padding:35px;
    text-align:center;
    margin-bottom:25px;
    box-shadow:0 8px 32px rgba(0,0,0,0.3);
}

.hero h1{
    color:white;
    font-size:50px;
    margin-bottom:10px;
}

.hero p{
    color:#d7f5dd;
    font-size:18px;
}

.glass{
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(15px);
    border:1px solid rgba(255,255,255,0.15);
    border-radius:20px;
    padding:20px;
    margin-bottom:20px;
    box-shadow:0 8px 32px rgba(0,0,0,0.25);
}

.upload-box{
    background: rgba(255,255,255,0.08);
    border-radius:20px;
    padding:20px;
    text-align:center;
}

h1,h2,h3,h4,h5,h6{
    color:white !important;
}

p,label{
    color:white !important;
}

[data-testid="stMetricValue"]{
    color:#7CFC00;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

def clean_name(name):
    name = name.replace("___", " ")
    name = name.replace("__", " ")
    name = name.replace("_", " ")
    return name.title()

st.sidebar.title("🌾 Agriculture Hub")

st.sidebar.markdown("""
### 🌱 Agriculture Facts

✅ Healthy crops increase yield

✅ Early disease detection saves money

✅ Proper irrigation prevents diseases

✅ Crop rotation improves soil health

✅ AI helps farmers detect diseases faster
""")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 🚜 Farmer Advice

• Inspect leaves regularly

• Remove infected plants quickly

• Avoid excessive watering

• Maintain proper spacing

• Use disease-free seeds
""")

st.sidebar.markdown("---")

st.sidebar.success(
    "Upload a leaf image to start AI analysis."
)

st.markdown("""
<div class="hero">

<h1>🌱 Smart Agriculture AI</h1>

<p>
AI Powered Crop Disease Detection Platform
<br>
Protecting Crops • Improving Yield • Helping Farmers
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass">

<h2>📤 Upload Crop Leaf Image</h2>

<p>
Upload a clear image of a crop leaf.
The AI model will analyze the image
and detect diseases with confidence scores.
</p>

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)
disease_info = {

    "Tomato_healthy": {
        "description": "Healthy tomato leaf.",
        "treatment": "No treatment required.",
        "prevention": "Maintain proper irrigation and nutrition."
    },

    "Potato___healthy": {
        "description": "Healthy potato leaf.",
        "treatment": "No treatment required.",
        "prevention": "Continue normal cultivation practices."
    },

    "Pepper__bell___healthy": {
        "description": "Healthy pepper leaf.",
        "treatment": "No treatment required.",
        "prevention": "Monitor regularly for pests and diseases."
    },

    "Tomato_Early_blight": {
        "description": "Fungal disease causing brown spots.",
        "treatment": "Apply fungicides and remove infected leaves.",
        "prevention": "Avoid overhead watering and rotate crops."
    },

    "Tomato_Late_blight": {
        "description": "Serious fungal infection spreading rapidly.",
        "treatment": "Use copper fungicides immediately.",
        "prevention": "Remove infected plants and improve airflow."
    },

    "Potato___Early_blight": {
        "description": "Dark circular spots on potato leaves.",
        "treatment": "Apply fungicide.",
        "prevention": "Keep foliage dry."
    },

    "Potato___Late_blight": {
        "description": "Water-soaked lesions turning brown.",
        "treatment": "Remove infected leaves.",
        "prevention": "Avoid excess moisture."
    },

    "Tomato_Bacterial_spot": {
        "description": "Small dark bacterial lesions.",
        "treatment": "Copper sprays.",
        "prevention": "Use disease-free seeds."
    },

    "Tomato_Leaf_Mold": {
        "description": "Yellow patches and mold growth.",
        "treatment": "Improve airflow and use fungicides.",
        "prevention": "Reduce humidity around plants."
    },

    "Tomato_Septoria_leaf_spot": {
        "description": "Circular spots with gray centers.",
        "treatment": "Remove infected leaves.",
        "prevention": "Avoid wet foliage."
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Damage caused by spider mites.",
        "treatment": "Use neem oil.",
        "prevention": "Monitor plants regularly."
    },

    "Tomato__Target_Spot": {
        "description": "Brown target-like lesions.",
        "treatment": "Apply fungicides.",
        "prevention": "Maintain field hygiene."
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Virus causing yellow curled leaves.",
        "treatment": "Control whiteflies.",
        "prevention": "Use resistant varieties."
    },

    "Tomato__Tomato_mosaic_virus": {
        "description": "Mosaic pattern on leaves.",
        "treatment": "Remove infected plants.",
        "prevention": "Disinfect tools regularly."
    },

    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial infection causing spots.",
        "treatment": "Copper sprays.",
        "prevention": "Avoid working with wet plants."
    }
}
def create_pdf(
    disease,
    confidence,
    description,
    treatment,
    prevention
):

    pdf = SimpleDocTemplate("report.pdf")

    styles = getSampleStyleSheet()

    content = [

        Paragraph(
            "Plant Disease Detection Report",
            styles["Title"]
        ),

        Paragraph(
            f"Disease: {disease}",
            styles["Normal"]
        ),

        Paragraph(
            f"Confidence: {confidence:.2f}%",
            styles["Normal"]
        ),

        Paragraph(
            f"Description: {description}",
            styles["Normal"]
        ),

        Paragraph(
            f"Treatment: {treatment}",
            styles["Normal"]
        ),

        Paragraph(
            f"Prevention: {prevention}",
            styles["Normal"]
        )
    ]

    pdf.build(content)

    return "report.pdf"

if uploaded_file is not None:

    col1, col2 = st.columns([1, 1])

    img = Image.open(uploaded_file).convert("RGB")

    with col1:

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.subheader("📷 Uploaded Image")

        st.image(img, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    img_resized = img.resize((224, 224))

    img_array = image.img_to_array(img_resized)

    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions)]

    confidence = float(np.max(predictions) * 100)

    info = disease_info.get(
        predicted_class,
        {
            "description": "No information available.",
            "treatment": "Consult an agricultural expert.",
            "prevention": "Monitor the crop regularly."
        }
    )

    with col2:

        st.markdown('<div class="glass">', unsafe_allow_html=True)

        st.subheader("🩺 Disease Analysis Report")

        if "healthy" in predicted_class.lower():

            st.success("🟢 Healthy Plant")

        else:

            st.error("🔴 Disease Detected")

        st.markdown(f"## {clean_name(predicted_class)}")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

        st.markdown("### 🦠 Description")
        st.write(info.get("description", "No information available."))

        st.markdown("### 💊 Treatment")
        st.write(info.get("treatment", "Consult an agricultural expert."))

        st.markdown("### 🛡 Prevention")
        st.write(info.get("prevention", "Monitor the crop regularly."))

        pdf_file = create_pdf(
            clean_name(predicted_class),
            confidence,
            info.get("description", "No information available."),
            info.get("treatment", "Consult an agricultural expert."),
            info.get("prevention", "Monitor the crop regularly.")
        )

        with open(pdf_file, "rb") as f:

            st.download_button(
                "📄 Download Report",
                f,
                file_name="plant_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## 📊 Top 3 Predictions")

    top_indices = np.argsort(predictions[0])[::-1][:3]

    for idx in top_indices:

        score = float(predictions[0][idx] * 100)

        st.markdown(f"### {clean_name(class_names[idx])}")

        st.progress(score / 100)

        st.write(f"{score:.2f}%")

    st.markdown("---")

    st.markdown("## 📈 Project Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Classes", "15")
    c2.metric("Images", "20,638")
    c3.metric("Accuracy", "87.13%")
    c4.metric("Model", "MobileNetV2")

    st.markdown("---")

    st.markdown("""
    <div class="glass">

    <h2>🌾 Daily Farmer Tips</h2>

    ✅ Check leaves every morning

    <br><br>

    ✅ Remove infected plants quickly

    <br><br>

    ✅ Avoid overwatering

    <br><br>

    ✅ Maintain proper spacing

    <br><br>

    ✅ Use certified seeds

    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div class="glass">

<h3>📌 About This System</h3>

This AI-powered agriculture platform uses
Transfer Learning with MobileNetV2 and
TensorFlow to identify plant diseases
from leaf images.

Dataset: PlantVillage

Classes: 15

</div>
""", unsafe_allow_html=True)

st.caption(
    "🌱 Smart Agriculture AI • Powered by TensorFlow & Streamlit | Created by Bhargav😉...."
)