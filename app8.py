from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
from reportlab.pdfgen import canvas
from datetime import datetime
import os
import sqlite3

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("performance_model.pkl")

# Define the features expected in the input
FEATURES = ["Hours Studied", "Previous Scores_scaled", "Extracurricular Activities",
            "Sleep Hours", "Sample Question Papers Practiced"]

# Create a directory to store PDF reports if it doesn't exist
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Initialize the SQLite database
def init_db():
    conn = sqlite3.connect("student_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS student_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            hours_studied REAL,
            previous_scores_scaled REAL,
            extracurricular_activities REAL,
            sleep_hours REAL,
            sample_question_papers_practiced REAL,
            predicted_class INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def home():
    """Render the index page where users can input data."""
    return render_template('index5.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Initialize data dictionary
        data = {}

        # Check if form data is received
        if request.form:
            data = request.form.to_dict()
            # Convert inputs to float, handle conversion errors
            try:
                data_float = {key: float(value) for key, value in data.items() if key != "Name"}
            except ValueError:
                return "Invalid input, unable to convert to float", 400

        # Ensure all required features are present
        if not all(feature in data_float for feature in FEATURES):
            return f"Missing one or more required features: {FEATURES}", 400

        # Get student's name (if provided)
        student_name = request.form.get("Name", "Student")

        # Create a DataFrame for the input
        input_data = pd.DataFrame([data_float])

        # Make predictions
        prediction = model.predict(input_data)
        predicted_class = int(prediction[0])

        # Save data to the database
        save_to_db(student_name, data_float, predicted_class)

        # Generate detailed guidance based on the prediction
        guidance = generate_guidance(predicted_class)

        # Generate PDF report
        pdf_filename = generate_pdf(student_name, guidance, predicted_class)

        # Return the generated PDF file for download
        return send_file(pdf_filename, as_attachment=True)

    except Exception as e:
        return f"Error: {str(e)}", 500

def save_to_db(name, data, predicted_class):
    """Save user input and prediction to the SQLite database."""
    conn = sqlite3.connect("student_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO student_entries (
            name, hours_studied, previous_scores_scaled, extracurricular_activities,
            sleep_hours, sample_question_papers_practiced, predicted_class
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        name,
        data.get("Hours Studied"),
        data.get("Previous Scores_scaled"),
        data.get("Extracurricular Activities"),
        data.get("Sleep Hours"),
        data.get("Sample Question Papers Practiced"),
        predicted_class
    ))
    conn.commit()
    conn.close()

def generate_guidance(predicted_class):
    """Generate personalized guidance based on the predicted class."""
    guidance_map = {
      1: (
            "Focus on improving your study habits. Allocate consistent hours daily and review your notes regularly.\n"
            "Here are some actionable steps:\n"
            "- Create a structured daily schedule for study and breaks.\n"
            "- Join study groups to enhance collaborative learning.\n"
            "- Seek help from teachers or tutors for difficult topics.\n"
            "- Reduce distractions by setting up a dedicated study space."
        ),
        2: (
            "Good work! Keep practicing additional sample papers to strengthen your preparation.\n"
            "Additional tips for further improvement:\n"
            "- Review mistakes from previous tests to avoid repeating them.\n"
            "- Increase focus on time management during tests.\n"
            "- Explore advanced resources to challenge yourself.\n"
            "- Maintain a healthy balance between study and relaxation."
        ),
        3: (
            "Excellent performance! Consider helping peers to reinforce your understanding and improve leadership skills.\n"
            "Suggestions to excel further:\n"
            "- Take on challenging projects or competitions.\n"
            "- Mentor others to solidify your knowledge.\n"
            "- Stay updated with new developments in your field of interest.\n"
            "- Keep a journal to track your progress and set new goals."
        ),
    }
    return guidance_map.get(predicted_class, "Maintain a balanced approach to your studies and extracurricular activities.")

def generate_pdf(student_name, guidance, predicted_class):
    """Generate a detailed PDF report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = os.path.join(REPORTS_DIR, f"student_report_{timestamp}.pdf")

    c = canvas.Canvas(pdf_filename)

    # Add content to the PDF
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, f"Student Performance Report for {student_name}")

    c.setFont("Helvetica", 12)
    c.drawString(100, 700, f"Predicted Performance Level: {predicted_class}")

    c.drawString(100, 650, "Personalized Guidance:")
    c.drawString(120, 630, guidance)

    c.drawString(100, 580, "Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    c.save()

    return pdf_filename

if __name__ == "__main__":
    app.run(debug=True)
