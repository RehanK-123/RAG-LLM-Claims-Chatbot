from flask import Flask, render_template, request, redirect, url_for, session
from backend import run_user_query

app = Flask(__name__)
app.secret_key = "supersecretkey"   # Needed for storing session data

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/auth", methods=["GET", "POST"])
def auth():
    if request.method == "POST":
        username = request.form.get("username")

        if username.strip():
            session["user"] = username
            session["history"] = []
            return redirect(url_for("chat"))

    return render_template("auth.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        user_msg = request.form["message"]

        # Add user message to chat history
        session["history"].append({"role": "user", "text": user_msg})

        # Run your AI logic
        bot_response = run_user_query(user_msg)

        # Add bot response to chat history
        session["history"].append({"role": "bot", "text": bot_response})

        session.modified = True

    return render_template("chat.html", user= session["user"], history=session["history"])
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)

