import os
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, session, flash
from .auth import create_user, verify_user, get_user
from .ml.model import (
    load_dataset,
    save_uploaded_dataset,
    is_dataset_uploaded,
    feature_selection_placeholder,
    train_and_evaluate,
    predict_text,
    get_training_stats,
)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.environ.get("APP_SECRET", "dev-secret-key")

    @app.context_processor
    def inject_dataset_status():
        return {"dataset_uploaded": is_dataset_uploaded()}

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            contact = request.form.get("contact", "")
            email = request.form.get("email", "")
            address = request.form.get("address", "")
            if not username or not password:
                flash("Username and password are required", "error")
            elif create_user(username, password, contact, email, address):
                flash("Signup successful. Please log in.", "success")
                return redirect(url_for("login"))
            else:
                flash("Username already exists", "error")
        return render_template("signup.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            if verify_user(username, password):
                session["username"] = username
                return redirect(url_for("dashboard"))
            flash("Invalid credentials", "error")
        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("index"))

    def require_login():
        if "username" not in session:
            return redirect(url_for("login"))
        return None

    @app.route("/dashboard")
    def dashboard():
        if (redir := require_login()) is not None:
            return redir
        return render_template("dashboard.html", username=session.get("username"))

    @app.route("/load", methods=["GET", "POST"])
    def load_fake_news():
        if (redir := require_login()) is not None:
            return redir
        
        records = []
        if request.method == "POST":
            # Check if a file was uploaded
            if 'csv_file' not in request.files:
                flash("No file selected", "error")
                return render_template("load.html", records=records)
            
            file = request.files['csv_file']
            
            # Check if file has a filename
            if file.filename == '':
                flash("No file selected", "error")
                return render_template("load.html", records=records)
            
            # Check if file is CSV
            if not file.filename.endswith('.csv'):
                flash("Please upload a CSV file", "error")
                return render_template("load.html", records=records)
            
            try:
                # Read the CSV file
                import csv
                from io import StringIO
                
                # Read file content as text
                stream = StringIO(file.stream.read().decode("UTF8"), newline=None)
                csv_reader = csv.DictReader(stream)
                
                # Convert to list of dictionaries
                for row in csv_reader:
                    image_path = row.get("image-path") or row.get("image_path") or ""
                    records.append({
                        "news": row.get("news", ""),
                        "target": row.get("target", ""),
                        "image_path": image_path
                    })
                
                # Save uploaded dataset so feature selection uses it
                save_uploaded_dataset(records)
                
                flash(f"Successfully loaded {len(records)} records from CSV file", "success")
                return render_template("load.html", records=records[:50])
            
            except Exception as e:
                flash(f"Error reading CSV file: {str(e)}", "error")
                return render_template("load.html", records=records)
        
        # GET request - show empty upload form
        return render_template("load.html", records=[])

    @app.route("/features")
    def features():
        if (redir := require_login()) is not None:
            return redir
        if not is_dataset_uploaded():
            flash("Please upload a dataset first before running feature selection.", "error")
            return redirect(url_for("load_fake_news"))
        stats = get_training_stats()
        return render_template("train.html", stats=stats)

    @app.route("/train")
    def train():
        if (redir := require_login()) is not None:
            return redir
        if not is_dataset_uploaded():
            flash("Please upload a dataset first before running the algorithm.", "error")
            return redirect(url_for("load_fake_news"))
        metrics, cm_img, perf_img = train_and_evaluate()
        cm_b64 = base64.b64encode(cm_img.getvalue()).decode("utf-8")
        perf_b64 = base64.b64encode(perf_img.getvalue()).decode("utf-8")
        return render_template("evaluate.html", metrics=metrics, cm_b64=cm_b64, perf_b64=perf_b64)

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        if (redir := require_login()) is not None:
            return redir
        result = None
        if request.method == "POST":
            text = request.form.get("text", "")
            if text.strip():
                label, prob = predict_text(text)
                result = {"label": label, "prob": f"{prob*100:.2f}%"}
        return render_template("predict.html", result=result)

    return app
