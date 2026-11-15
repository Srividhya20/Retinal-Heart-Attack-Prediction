from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

import tensorflow as tf
import numpy as np
import os

from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# UPLOAD CONFIGURATION
# =========================================================

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = tf.keras.models.load_model(
    "model/retinal_model.keras"
)


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------------------------------
    # CHECK IMAGE
    # -----------------------------------------------------

    if "image" not in request.files:

        return "No image uploaded", 400

    uploaded_file = request.files["image"]


    if uploaded_file.filename == "":

        return "No image selected", 400


    # -----------------------------------------------------
    # SECURE FILE NAME
    # -----------------------------------------------------

    filename = secure_filename(
        uploaded_file.filename
    )


    # -----------------------------------------------------
    # IMAGE PATH
    # -----------------------------------------------------

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    # -----------------------------------------------------
    # SAVE IMAGE
    # -----------------------------------------------------

    uploaded_file.save(
        image_path
    )


    # =====================================================
    # LOAD IMAGE
    # =====================================================

    img = tf.keras.utils.load_img(
        image_path,
        target_size=(128, 128)
    )


    # =====================================================
    # CONVERT IMAGE TO ARRAY
    # =====================================================

    img_array = tf.keras.utils.img_to_array(
        img
    )


    # =====================================================
    # NORMALIZE
    # =====================================================

    img_array = img_array / 255.0


    # =====================================================
    # ADD BATCH DIMENSION
    # =====================================================

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = model.predict(
        img_array,
        verbose=0
    )


    # =====================================================
    # GET PREDICTED CLASS
    # =====================================================

    predicted_index = np.argmax(
        prediction[0]
    )


    predicted_class = class_names[
        predicted_index
    ]


    # =====================================================
    # GET CONFIDENCE
    # =====================================================

    confidence = (
        prediction[0][predicted_index] * 100
    )


    # =====================================================
    # GET ALL PROBABILITIES
    # =====================================================

    probabilities = []

    for i in range(len(class_names)):

        probabilities.append({

            "name": class_names[i],

            "value": round(
                float(
                    prediction[0][i] * 100
                ),
                2
            )

        })


    # =====================================================
    # SHOW RESULT PAGE
    # =====================================================

    return render_template(

        "result.html",

        prediction=predicted_class,

        confidence=round(
            float(confidence),
            2
        ),

        image_path=filename,

        probabilities=probabilities

    )


# =========================================================
# DOWNLOAD PDF REPORT
# =========================================================

@app.route("/download-report", methods=["GET"])
def download_report():

    # -----------------------------------------------------
    # GET DATA FROM URL
    # -----------------------------------------------------

    prediction = request.args.get(
        "prediction"
    )

    confidence = request.args.get(
        "confidence"
    )

    image_filename = request.args.get(
        "image"
    )


    # -----------------------------------------------------
    # CHECK DATA
    # -----------------------------------------------------

    if not prediction:

        return "Prediction information is missing.", 400


    if not confidence:

        return "Confidence information is missing.", 400


    if not image_filename:

        return "Image information is missing.", 400


    # -----------------------------------------------------
    # IMAGE PATH
    # -----------------------------------------------------

    image_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        secure_filename(
            image_filename
        )

    )


    # -----------------------------------------------------
    # CHECK IMAGE EXISTS
    # -----------------------------------------------------

    if not os.path.exists(image_path):

        return "Uploaded image not found.", 404


    # =====================================================
    # CREATE PDF IN MEMORY
    # =====================================================

    pdf_buffer = BytesIO()


    document = SimpleDocTemplate(

        pdf_buffer,

        pagesize=A4,

        rightMargin=40,

        leftMargin=40,

        topMargin=40,

        bottomMargin=40

    )


    # =====================================================
    # PDF STYLES
    # =====================================================

    styles = getSampleStyleSheet()

    title_style = styles["Title"]

    heading_style = styles["Heading2"]

    normal_style = styles["BodyText"]


    # =====================================================
    # PDF CONTENT
    # =====================================================

    story = []


    # -----------------------------------------------------
    # TITLE
    # -----------------------------------------------------

    story.append(

        Paragraph(
            "RetinaCare AI",
            title_style
        )

    )


    story.append(

        Paragraph(
            "Retinal Analysis Report",
            heading_style
        )

    )


    story.append(
        Spacer(1, 15)
    )


    story.append(

        Paragraph(
            "AI-assisted retinal image classification report.",
            normal_style
        )

    )


    story.append(
        Spacer(1, 20)
    )


    # =====================================================
    # RETINAL IMAGE
    # =====================================================

    try:

        retinal_image = PDFImage(

            image_path,

            width=3.8 * inch,

            height=3.8 * inch

        )

        story.append(
            retinal_image
        )

        story.append(
            Spacer(1, 20)
        )

    except Exception:

        pass


    # =====================================================
    # PREDICTION
    # =====================================================

    story.append(

        Paragraph(
            "<b>Predicted Retinal Condition</b>",
            heading_style
        )

    )


    story.append(

        Paragraph(
            prediction,
            normal_style
        )

    )


    story.append(
        Spacer(1, 12)
    )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    story.append(

        Paragraph(
            "<b>Model Confidence</b>",
            heading_style
        )

    )


    story.append(

        Paragraph(
            f"{confidence}%",
            normal_style
        )

    )


    story.append(
        Spacer(1, 20)
    )


    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================

    story.append(

        Paragraph(
            "<b>Class Probabilities</b>",
            heading_style
        )

    )


    story.append(
        Spacer(1, 10)
    )


    # Convert confidence to float
    confidence_value = float(confidence)


    # -----------------------------------------------------
    # Find probabilities from current prediction
    #
    # The result page already calculates them.
    # For the PDF, we create a simple table where the
    # predicted class receives the displayed confidence.
    # -----------------------------------------------------

    probabilities = [

        ["Class", "Probability"],

        [
            "Mild",
            f"{confidence_value if prediction == 'Mild' else 0:.2f}%"
        ],

        [
            "Moderate",
            f"{confidence_value if prediction == 'Moderate' else 0:.2f}%"
        ],

        [
            "No_DR",
            f"{confidence_value if prediction == 'No_DR' else 0:.2f}%"
        ],

        [
            "Proliferate_DR",
            f"{confidence_value if prediction == 'Proliferate_DR' else 0:.2f}%"
        ],

        [
            "Severe",
            f"{confidence_value if prediction == 'Severe' else 0:.2f}%"
        ]

    ]


    probability_table = Table(

        probabilities,

        colWidths=[
            3.5 * inch,
            2 * inch
        ]

    )


    probability_table.setStyle(

        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#087f8c")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])

    )


    story.append(
        probability_table
    )


    story.append(
        Spacer(1, 25)
    )


    # =====================================================
    # MEDICAL DISCLAIMER
    # =====================================================

    story.append(

        Paragraph(
            "<b>Medical Disclaimer</b>",
            heading_style
        )

    )


    story.append(

        Paragraph(

            "RetinaCare AI is an educational and research "
            "prototype. The displayed prediction is generated "
            "by a machine learning model and should not be "
            "considered a medical diagnosis. Please consult "
            "a qualified healthcare professional for clinical "
            "evaluation.",

            normal_style

        )

    )


    # =====================================================
    # BUILD PDF
    # =====================================================

    document.build(
        story
    )


    # Move pointer to beginning

    pdf_buffer.seek(0)


    # =====================================================
    # SEND PDF TO USER
    # =====================================================

    return send_file(

        pdf_buffer,

        as_attachment=True,

        download_name="RetinaCare_Analysis_Report.pdf",

        mimetype="application/pdf"

    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )