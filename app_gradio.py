# app_gradio_full.py
import os
import io
import tempfile
import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageStat, ImageFilter
import matplotlib.pyplot as plt
from fpdf import FPDF
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import pyttsx3

# -----------------------
# Config / Paths
# -----------------------
MODEL_PATH = r"C:\Skin_cencer_detection\models\skin_cancer_model.keras"
INPUT_SIZE = (224, 224)

# Demo class mapping (change to real classes)
MULTI_CLASSES = ['Benign Nevus', 'Actinic Keratosis', 'Basal Cell Carcinoma', 'Melanoma']
CLASS_MAPPING = {i: c for i, c in enumerate(MULTI_CLASSES)}

# Credentials (example)
DOCTOR_USERS = {"doctor": "password123", "admin": "securepass"}
PATIENT_USERS = {"patient1": "pass", "patient2": "pass"}

# In-memory DBs (for demo; persist to CSV/db for real)
appointments = pd.DataFrame(columns=["id","patient_name","date","time","doctor","notes"])
history_data = pd.DataFrame(columns=['Time', 'Patient ID', 'Result', 'Confidence', 'Risk'])
users_session = {"user_role":"guest", "username":None, "id": None, "name":None, "city":""}

# Try load model if present
MODEL_LOADED = False
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH, compile=False)
        MODEL_LOADED = True
        print("Model loaded.")
    except Exception as e:
        print("Model exists but failed to load:", e)
        model = None
else:
    model = None
    print("Model file not found; running in simulated mode.")

# Setup TTS engine for voice assistant (offline)
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)

# -----------------------
# Utility functions
# -----------------------
def login(username, password):
    if username in DOCTOR_USERS and DOCTOR_USERS[username] == password:
        users_session.update({"user_role":"doctor","username":username})
        return "Doctor login successful.", gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    elif username in PATIENT_USERS and PATIENT_USERS[username] == password:
        users_session.update({"user_role":"patient","username":username})
        return "Patient login successful.", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    else:
        users_session.update({"user_role":"guest","username":None})
        return "Login failed: invalid credentials.", gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def update_patient_info(patient_name, age, gender, city, history_problems):
    pid = users_session.get("username", f"PAT-{np.random.randint(1000,9999)}")
    users_session.update({"id":pid, "name":patient_name, "age":age, "gender":gender, "city":city, "history":history_problems})
    return f"Saved: {patient_name} | Patient ID: {pid}"

def check_image_quality(pil_img: Image.Image):
    """Simple quality checks: size, blurriness (via variance of laplacian approx), brightness"""
    try:
        img = pil_img.convert("RGB")
        # size check
        w,h = img.size
        size_ok = (w >= 224 and h >= 224)
        # brightness
        stat = ImageStat.Stat(img)
        brightness = sum(stat.mean)/3
        brightness_ok = (30 < brightness < 220)
        # blur check: use variance of Laplacian like approach via edge detection
        gray = img.convert("L")
        lap = gray.filter(ImageFilter.FIND_EDGES)
        var = np.var(np.array(lap))
        blur_ok = var > 50  # heuristic
        return {
            "size_ok": size_ok, "brightness": float(brightness),
            "brightness_ok": brightness_ok, "focus_var": float(var), "blur_ok": blur_ok
        }
    except Exception as e:
        return {"error": str(e)}

def generate_probability_graph(probs):
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(MULTI_CLASSES, probs)
    ax.set_ylim(0,1)
    plt.xticks(rotation=30, ha="right")
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def save_pdf_report(patient_info, result_text, graph_buf, heatmap_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0,10, "Skin Cancer Report", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=10)
    pdf.cell(0,8, f"Patient: {patient_info.get('name','N/A')} (ID: {patient_info.get('id','N/A')})", ln=True)
    pdf.cell(0,8, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)
    pdf.multi_cell(0,6, f"Result:\n{result_text}")
    pdf.ln(4)
    # add graph
    tmpg = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmpg.write(graph_buf.getvalue())
    tmpg.flush()
    pdf.image(tmpg.name, w=120)
    # add heatmap as small image
    if heatmap_img:
        tmph = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        heatmap_img.save(tmph.name)
        pdf.ln(4)
        pdf.image(tmph.name, w=60)
    out_path = os.path.join(tempfile.gettempdir(), f"skin_report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
    pdf.output(out_path)
    return out_path

def get_gradcam_heatmap(model, img_array):
    # basic gradcam fallback: return gray heatmap or placeholder
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv = layer.name
                break
        if last_conv is None:
            return Image.new("RGB", INPUT_SIZE, (255,180,180))
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv).output, model.output])
        with tf.GradientTape() as tape:
            conv_outs, preds = grad_model(tf.cast(img_array, tf.float32))
            loss = preds[:, tf.argmax(preds[0])]
        grads = tape.gradient(loss, conv_outs)
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        conv = conv_outs[0]
        heatmap = tf.reduce_sum(conv * pooled, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-8
        heatmap = np.uint8(255 * heatmap.numpy())
        img = Image.fromarray(heatmap).resize(INPUT_SIZE).convert("RGB")
        return img
    except Exception as e:
        return Image.new("RGB", INPUT_SIZE, (255,180,180))

def text_to_speech(text):
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts_engine.save_to_file(text, tmp.name)
        tts_engine.runAndWait()
        return tmp.name
    except Exception as e:
        return None

# -----------------------
# Core prediction handler
# -----------------------
def predict_and_report(pil_img: Image.Image, language='English', symptoms_text=""):
    # access control
    if users_session.get("user_role") != "doctor":
        return "Access Denied: Only doctors can run analysis.", None, None, None, history_data

    # image quality check
    q = check_image_quality(pil_img)

    # preprocess
    img = pil_img.resize(INPUT_SIZE).convert("RGB")
    arr = img_to_array(img)/255.0
    arr_exp = np.expand_dims(arr, 0)

    # predict (real model if loaded else simulated multiclass)
    if MODEL_LOADED and model is not None:
        preds = model.predict(arr_exp)
        # If binary model: adapt. Here we try to handle both shapes.
        if preds.shape[-1] == 1:
            p = float(preds[0][0])
            if p >= 0.5:
                probs = np.array([0.05,0.10,0.35,0.50])
            else:
                probs = np.array([0.70,0.15,0.10,0.05])
        else:
            probs = preds[0]
            # if model returns logits, softmax
            if (probs.max() > 1.0) or (probs.min() < 0):
                probs = tf.nn.softmax(probs).numpy()
    else:
        # simulated
        p = 0.7
        probs = np.array([0.05,0.10,0.35,0.50]) if p>=0.5 else np.array([0.70,0.15,0.10,0.05])

    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_MAPPING.get(predicted_idx, "Unknown")
    confidence = float(probs[predicted_idx])

    # graph
    graph_buf = generate_probability_graph(probs)

    # heatmap
    heatmap_img = get_gradcam_heatmap(model, arr_exp) if MODEL_LOADED else Image.new("RGB", INPUT_SIZE, (255,180,180))
    overlay = Image.blend(img, heatmap_img, alpha=0.5)

    # compose result text with symptoms-based risk (very simple rules)
    risk = "High" if confidence >= 0.8 else "Medium"
    symptom_flag = False
    if symptoms_text:
        low = ["itch", "small", "slow"]
        high = ["bleeding","changing","rapid","pain"]
        text_low = symptoms_text.lower()
        if any(w in text_low for w in high):
            symptom_flag = True
            risk = "High"
    result_text = f"{'निष्कर्ष:' if language=='Hindi' else 'Diagnosis:'} {predicted_class} | Confidence: {confidence:.2f}\nRisk: {risk}"
    if q.get("error"):
        result_text += f"\nImage QC error: {q['error']}"
    else:
        result_text += f"\nImage QC -> size_ok:{q['size_ok']} brightness:{q['brightness']:.1f} blur_ok:{q['blur_ok']}"

    # update history
    history_data.loc[len(history_data)] = [
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        users_session.get("id","N/A"),
        predicted_class,
        f"{confidence:.2f}",
        risk
    ]

    # save pdf
    pdf_path = save_pdf_report(users_session, result_text, graph_buf, heatmap_img)

    return result_text, Image.open(graph_buf), overlay, history_data, pdf_path

# -----------------------
# Appointments handlers
# -----------------------
def book_appointment(patient_name, date_str, time_str, doctor, notes):
    appt_id = f"APT-{np.random.randint(1000,9999)}"
    row = {"id":appt_id, "patient_name":patient_name, "date":date_str, "time":time_str, "doctor":doctor, "notes":notes}
    global appointments
    appointments.loc[len(appointments)] = row
    return f"Appointment booked: {appt_id}", appointments

def cancel_appointment(appt_id):
    global appointments
    mask = appointments['id'] != appt_id
    if mask.sum()==len(appointments):
        return "Appointment ID not found.", appointments
    appointments = appointments[mask].reset_index(drop=True)
    return f"Cancelled {appt_id}", appointments

def download_appointments_csv():
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    appointments.to_csv(tmp.name, index=False)
    return tmp.name

# -----------------------
# Voice assistant
# -----------------------
def speak_text(text):
    path = text_to_speech(text)
    if path:
        return f"Saved audio: {path}", path
    else:
        return "TTS failed", None

# -----------------------
# Gradio UI
# -----------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🩻 Skin Cancer System — Full Demo")
    # top row: login
    with gr.Row():
        with gr.Column(scale=2):
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_status = gr.Textbox(value="Please login", interactive=False)
        with gr.Column(scale=3):
            gr.Markdown("**Session:**")
            sess_info = gr.JSON(value=users_session)

    # Tabs area
    with gr.Tabs() as tabs:
        with gr.TabItem("Patient Intake"):
            name = gr.Textbox(label="Name")
            age = gr.Number(label="Age")
            gender = gr.Radio(["Male","Female","Other"], label="Gender")
            city = gr.Textbox(label="City")
            hist = gr.Textbox(label="History")
            save_btn = gr.Button("Save Patient")
            pid_out = gr.Textbox(label="Patient Info", interactive=False)

        with gr.TabItem("Diagnosis & Report"):
            img_in = gr.Image(type="pil", label="Upload lesion image")
            symptoms = gr.Textbox(label="Symptoms (optional)")
            lang = gr.Radio(["English","Hindi"], value="English", label="Language")
            run_btn = gr.Button("Analyze (Doctors only)")
            out_text = gr.Textbox(label="Result", lines=6)
            out_graph = gr.Image(label="Probability Graph")
            out_heat = gr.Image(label="Grad-CAM Overlay")
            out_history = gr.Dataframe(value=history_data)
            pdf_download = gr.File(label="Download Report")

        with gr.TabItem("Appointments"):
            ap_name = gr.Textbox(label="Patient Name")
            ap_date = gr.Date(label="Date")
            ap_time = gr.Textbox(label="Time (e.g. 10:30)")
            ap_doctor = gr.Textbox(label="Doctor")
            ap_notes = gr.Textbox(label="Notes")
            book_btn = gr.Button("Book")
            book_out = gr.Textbox()
            ap_table = gr.Dataframe(value=appointments)
            cancel_id = gr.Textbox(label="Appointment ID to cancel")
            cancel_btn = gr.Button("Cancel")
            download_btn = gr.Button("Download CSV")

        with gr.TabItem("Doctor Dashboard"):
            gr.Markdown("## Dashboard (Doctors only)")
            hist_table = gr.Dataframe(value=history_data)
            download_hist = gr.Button("Download History CSV")

        with gr.TabItem("Voice Assistant"):
            tts_text = gr.Textbox(label="Text to speak")
            tts_button = gr.Button("Speak & Save")
            tts_out = gr.Textbox()
            tts_file = gr.File()

    # Bind buttons
    login_btn.click(login, inputs=[username,password], outputs=[login_status, gr.update(), gr.update(), gr.update()])
    save_btn.click(update_patient_info, inputs=[name, age, gender, city, hist], outputs=[pid_out])
    run_btn.click(predict_and_report, inputs=[img_in, lang, symptoms], outputs=[out_text, out_graph, out_heat, out_history, pdf_download])
    book_btn.click(book_appointment, inputs=[ap_name, ap_date, ap_time, ap_doctor, ap_notes], outputs=[book_out, ap_table])
    cancel_btn.click(cancel_appointment, inputs=[cancel_id], outputs=[book_out, ap_table])
    download_btn.click(lambda: download_appointments_csv(), outputs=[gr.File()])
    download_hist.click(lambda: (tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name), outputs=[gr.File()])
    tts_button.click(speak_text, inputs=[tts_text], outputs=[tts_out, tts_file])

if __name__ == "__main__":
    demo.launch()
