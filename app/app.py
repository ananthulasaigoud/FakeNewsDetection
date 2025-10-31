import os
import io
import base64
from flask import Flask, render_template, request, redirect, url_for, session, flash
from .auth import create_user, verify_user, get_user
from .ml.model import (
    load_dataset,
    feature_selection_placeholder,
    train_and_evaluate,
    predict_text,
    get_training_stats,
)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.environ.get("APP_SECRET", "dev-secret-key")

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

    @app.route("/load")
    def load_fake_news():
        if (redir := require_login()) is not None:
            return redir
        records = load_dataset()
        return render_template("load.html", records=records[:50])

    @app.route("/features")
    def features():
        if (redir := require_login()) is not None:
            return redir
        stats = get_training_stats()
        return render_template("train.html", stats=stats)

    @app.route("/train")
    def train():
        if (redir := require_login()) is not None:
            return redir
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
