import cv2
import threading
from flask import Flask, url_for, render_template, request, redirect, session, g, flash, get_flashed_messages
from werkzeug.security import check_password_hash, generate_password_hash
import sqlite3
from functools import wraps

from face_detection import detect_faces

app = Flask(__name__)
app.config["SECRET_KEY"] = "secretkey"

camera = None  # Global variable to store the camera stream
stream_thread = None  # To track the OpenCV stream thread
streaming = False  # To check if the stream is active


def open_camera_stream(camera_url):
    """Function to open the camera stream in a new window using OpenCV."""
    global streaming
    streaming = True
    cap = cv2.VideoCapture(camera_url)

    if not cap.isOpened():
        print("Error: Unable to open camera stream.")
        streaming = False
        return

    # Bring the OpenCV window to the front
    cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Camera Stream", cv2.WND_PROP_TOPMOST, 1)

    frame_skip = 3  # Process every 3rd frame
    frame_count = 0

    while streaming:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to retrieve frame.")
            break

        frame_count += 1
        if frame_count % frame_skip == 0:
            # Detect faces only on every 'frame_skip' frame
            frame_with_faces = detect_faces(frame)
            cv2.imshow("Camera Stream", frame_with_faces)

        # Exit the stream when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    streaming = False

DATABASE = "db.sqlite3"

# Function to get the database connection
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE, check_same_thread=False)
        g.cursor = g.db.cursor()
    return g.db, g.cursor

# Function to close the database connection
@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Initialize the database
def initialize_database():
    db, c = get_db()

    # Create the 'users' table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        user_type TEXT NOT NULL
    )
    ''')

    # Generate a hashed password
    hashed_org_password = generate_password_hash("org123", method="pbkdf2:sha256", salt_length=8)

    # Insert the user if they don't already exist
    c.execute('''
    INSERT OR IGNORE INTO users (email, password, user_type)
    VALUES (:email, :password, :user_type)
    ''', {
        "email": "org123@gmail.com",
        "password": hashed_org_password,
        "user_type": "organization"
    })

    db.commit()

# Initialize the database at application start
with app.app_context():
    initialize_database()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # If the user is not logged in, redirect to the login page
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/")
def home_page():
    flash("hiiiiiii", "success")
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    db, c = get_db();
    signupErrors = ''

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")
        # check if the form is valid

        if not email or not password or not confirmation:
            signupErrors = "please fill out all fields"

        if password != confirmation:
            signupErrors = "password confirmation doesn't match password"

        # check if email exist in the database
        exist = c.execute("SELECT * FROM users WHERE email=:email", {"email": email}).fetchall()

        if len(exist) != 0:
            signupErrors = "user already registered"
        else:
            # hash the password
            pwhash = generate_password_hash(password, method="pbkdf2:sha256", salt_length=8)

            # insert the row
            c.execute("INSERT INTO users (email, password, user_type) VALUES (:email, :password, :user_type)", {"email": email, "password": pwhash, "user_type": "employee"})
            db.commit()

            return render_template('login.html');
    
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    db, c = get_db();
    loginErrors = ''  # Initialize variable for error messages

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # Check if required fields are filled
        if not email or not password:
            loginErrors = "Please fill out all required fields."
        else:
            # Check if the email exists in the database
            user = c.execute("SELECT * FROM users WHERE email=:email", {"email": email}).fetchone()

            if not user:
                loginErrors = "Email not registered. Please register first."
            else:
                # Validate the password
                stored_password_hash = user[2]  # Assuming the password hash is stored in the third column
                if not check_password_hash(stored_password_hash, password):
                    loginErrors = "Invalid password. Please try again."
                else:
                    # Login the user
                    session["user_id"] = user[0]  # Assuming the user ID is in the first column
                    session["email"] = user[1]
                    session["user_type"] = user[3]

                    return redirect("/dashboard")  # Redirect to employee dashboard

    return render_template("login.html", loginErrors=loginErrors)

@app.route("/logout",  methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/about")
def about():
     return render_template("about.html")

@app.route("/how-to-use")
def howToUse():
    return render_template("how-to-use.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/dashboard")
@login_required
def userDashboard():
    if session['user_type'] == 'employee':
        return render_template("user-templates/user-dashboard.html")
    else:
        return render_template("org-templates/org-dashboard.html")

@app.route('/link-camera', methods=["GET", "POST"])
@login_required
def linkCamera():
    global camera, stream_thread, streaming

    camera_url = None

    if request.method == "POST":
        # Get the camera URL from the form
        camera_url = request.form.get("camera-url")

        # Start the OpenCV stream in a separate thread
        if not streaming:
            try:
                stream_thread = threading.Thread(target=open_camera_stream, args=(camera_url,))
                stream_thread.start()
                flash("Stream started successfully.", "success")
            except Exception as e:
                flash(f"Error starting stream: {e}", "error")
        else:
            flash("Stream is already running.", "warning")

    return render_template("org-templates/link-camera.html", streaming=streaming, camera_url=camera_url)


@app.route('/kill-stream', methods=["POST"])
def kill_stream():
    global streaming
    if streaming:
        streaming = False  # Stop the OpenCV stream
    return redirect(url_for('linkCamera'))

@app.route('/employee-attendance')
@login_required
def employeeAttendance():
    return render_template("org-templates/employee-attendance.html")

@app.route('/profile')
@login_required
def orgProfile():
    if session['user_type'] == 'employee':
        return render_template("user-templates/profile.html")
    else:
        return render_template("org-templates/profile.html")
