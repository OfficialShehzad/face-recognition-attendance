import os
import cv2
import threading
from flask import Flask, url_for, render_template, request, redirect, session, g, flash, get_flashed_messages
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import numpy as np
import sqlite3
from functools import wraps
import datetime
import pytz  # for timezone conversion
from zoneinfo import ZoneInfo  # For Python 3.9+

from live_stream_recognition import recognize_faces
from train_image import train_image

app = Flask(__name__)
app.config["SECRET_KEY"] = "secretkey"

# Directory to save uploaded files
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

DATABASE = "db.sqlite3"

haarcasecade_path = "haarcascade_frontalface_default.xml"
trainimagelabel_path = "./TrainingImageLabel/Trainner.yml"
trainimage_path = "/TrainingImage"
if not os.path.exists(trainimage_path):
    os.makedirs(trainimage_path)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

camera = None  # Global variable to store the camera stream
stream_thread = None  # To track the OpenCV stream thread
streaming = False  # To check if the stream is active


def open_camera_stream(camera_url):
    with app.app_context():
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
        cv2.imshow("Camera Stream", np.zeros((100, 100, 3), dtype=np.uint8))  # Show a dummy image
        cv2.waitKey(1)  # Give OpenCV some time to create the window
        cv2.setWindowProperty("Camera Stream", cv2.WND_PROP_TOPMOST, 1)

        frame_skip = 8  # Process every 8th frame
        frame_count = 0

        while streaming:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to retrieve frame.")
                break

            frame_count += 1
            if frame_count % frame_skip == 0:
                # Detect faces only on every 'frame_skip' frame
                # frame_with_faces = detect_faces(frame)
                frame_with_faces = recognize_faces(frame, get_db)
                cv2.imshow("Camera Stream", frame_with_faces)

            # Exit the stream when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        streaming = False

# Function to validate allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        user_name TEXT,
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
        "user_name": "Org Admin",
        "user_type": "organization"
    })

    # Create the 'userImages' table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS userImages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    ''')

    # Create the 'trainingStatus' table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS trainingStatus (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        is_trained BOOLEAN NOT NULL DEFAULT 0,
        trained_at TIMESTAMP DEFAULT NULL,
        FOREIGN KEY (image_id) REFERENCES userImages (id) ON DELETE CASCADE
    )
    ''')

    # Create attendance table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        status TEXT CHECK(status IN ('entered', 'exited')) NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    ''')


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
    
    return render_template("register.html", signupErrors = signupErrors)


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
                    session["user_name"] = user[3]
                    session["user_type"] = user[4]

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
    db, c = get_db()
    user_type = session['user_type']
    user_id = session['user_id']

    today = datetime.date.today()
    start_datetime = datetime.datetime.combine(today, datetime.datetime.min.time())
    end_datetime = datetime.datetime.combine(today, datetime.datetime.max.time())

    if user_type == 'employee':
        # Employee attendance
        c.execute("""
            SELECT DATE(timestamp), MIN(timestamp), MAX(timestamp)
            FROM attendance 
            WHERE user_id = ? 
            AND timestamp BETWEEN ? AND ?
        """, (user_id, start_datetime, end_datetime))
        record = c.fetchone()

        first_entry = record[1]
        last_exit = record[2]
        duration = None

        # Convert strings to datetime
        if first_entry and last_exit:
            first_entry_dt = datetime.datetime.fromisoformat(first_entry)
            last_exit_dt = datetime.datetime.fromisoformat(last_exit)
            diff = last_exit_dt - first_entry_dt
            duration = str(diff)

        return render_template("user-templates/user-dashboard.html", 
                               date=record[0], 
                               first_entry=first_entry, 
                               last_exit=last_exit, 
                               duration=duration)

    else:
        # Org dashboard
        c.execute("""
            SELECT u.user_name, MIN(a.timestamp), MAX(a.timestamp)
            FROM attendance a
            JOIN users u ON u.id = a.user_id
            WHERE a.timestamp BETWEEN ? AND ?
            GROUP BY u.user_name
        """, (start_datetime, end_datetime))
        records = c.fetchall()

        summary = []
        for user_name, first_entry, last_exit in records:
            duration = None
            if first_entry and last_exit:
                first_entry_dt = datetime.datetime.fromisoformat(first_entry)
                last_exit_dt = datetime.datetime.fromisoformat(last_exit)
                diff = last_exit_dt - first_entry_dt
                duration = str(diff)
            summary.append({
                'user_name': user_name,
                'first_entry': first_entry,
                'last_exit': last_exit,
                'duration': duration
            })

        c.execute("SELECT COUNT(*) FROM users WHERE user_type='employee'")
        total_employees = c.fetchone()[0]
        present_today = len(records)

        return render_template("org-templates/org-dashboard.html", 
                               summary=summary, 
                               total_employees=total_employees, 
                               present_today=present_today,
                               date=today)

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
    db, c = get_db()

    # Get selected date from query params
    date_str = request.args.get('date')
    if not date_str:
        # Default to today's date
        selected_date = datetime.datetime.now().date()
    else:
        selected_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

    # Prepare start and end datetime strings
    start_datetime = f"{selected_date} 00:00:00"
    end_datetime = f"{selected_date} 23:59:59"

    # Fetch records for the selected date
    c.execute("""
        SELECT u.user_name, DATE(a.timestamp) AS date, 
            MIN(CASE WHEN a.status = 'entered' THEN a.timestamp END) AS first_entry, 
            MAX(CASE WHEN a.status = 'exited' THEN a.timestamp END) AS last_exit
        FROM attendance a
        JOIN users u ON u.id = a.user_id
        WHERE a.timestamp BETWEEN ? AND ?
        GROUP BY u.user_name, date
        ORDER BY u.user_name ASC
    """, (start_datetime, end_datetime))
    records = c.fetchall()

    # Define your local timezone
    LOCAL_TZ = ZoneInfo('Asia/Kolkata')  # Replace with your local timezone

    attendance_summary = []
    for record in records:
        user_name, date_str, first_entry, last_exit = record

        # Convert timestamps
        if first_entry and last_exit:
            # Parse the UTC timestamp
            first_dt_utc = datetime.datetime.strptime(first_entry, '%Y-%m-%d %H:%M:%S')
            last_dt_utc = datetime.datetime.strptime(last_exit, '%Y-%m-%d %H:%M:%S')

            # Assume UTC timezone
            first_dt_utc = first_dt_utc.replace(tzinfo=ZoneInfo('UTC'))
            last_dt_utc = last_dt_utc.replace(tzinfo=ZoneInfo('UTC'))

            # Convert to local time
            first_dt_local = first_dt_utc.astimezone(LOCAL_TZ)
            last_dt_local = last_dt_utc.astimezone(LOCAL_TZ)

            # Calculate duration
            duration = last_dt_local - first_dt_local
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            duration_str = f"{hours}h {minutes}m {seconds}s"
        else:
            first_dt_local = last_dt_local = 'N/A'
            duration_str = 'N/A'

        attendance_summary.append({
            'user_name': user_name,
            'date': selected_date.strftime('%Y-%m-%d'),
            'first_entry': first_dt_local.strftime('%Y-%m-%d %H:%M:%S') if first_entry else 'N/A',
            'last_exit': last_dt_local.strftime('%Y-%m-%d %H:%M:%S') if last_exit else 'N/A',
            'total_duration': duration_str
        })

    return render_template(
        "org-templates/employee-attendance.html",
        attendance_summary=attendance_summary,
        selected_date=selected_date.strftime('%Y-%m-%d')
    )



@app.route('/my-attendance')
@login_required
def myAttendance():
    db, c = get_db()
    user_id = session['user_id']

    # Fetch attendance grouped by DATE
    c.execute("""
        SELECT 
            DATE(timestamp) AS date,
            MIN(CASE WHEN status = 'entered' THEN timestamp END) AS first_entry,
            MAX(CASE WHEN status = 'exited' THEN timestamp END) AS last_exit
        FROM attendance
        WHERE user_id = ?
        GROUP BY date
        ORDER BY date DESC
    """, (user_id,))

    
    records = c.fetchall()
    print(records)

    attendance_summary = []
    for record in records:
        date_str, first_entry, last_exit = record
        if first_entry and last_exit:
            # Convert to datetime objects
            first_dt = datetime.datetime.strptime(first_entry, '%Y-%m-%d %H:%M:%S')
            last_dt = datetime.datetime.strptime(last_exit, '%Y-%m-%d %H:%M:%S')

            # Assuming UTC in DB, convert to local timezone (example: IST)
            utc = pytz.utc
            local_tz = pytz.timezone('Asia/Kolkata')  # Change as per your location

            first_local = utc.localize(first_dt).astimezone(local_tz)
            last_local = utc.localize(last_dt).astimezone(local_tz)

            # Calculate duration
            duration = last_local - first_local

            attendance_summary.append({
                'date': first_local.strftime('%Y-%m-%d'),
                'first_entry': first_local.strftime('%I:%M %p'),
                'last_exit': last_local.strftime('%I:%M %p'),
                'duration': str(duration)
            })

    return render_template(
        "user-templates/my-attendance.html",
        attendance_summary=attendance_summary
    )


@app.route('/profile')
@login_required
def orgProfile():
    if session['user_type'] == 'employee':
        db, c = get_db()
        images = c.execute("SELECT image_path, id FROM userImages WHERE user_id = ?", (session['user_id'],)).fetchall()
        
        trained_images = c.execute(""
        "SELECT ui.id "
        "FROM trainingStatus ts "
        "JOIN userImages ui "
        "ON ts.image_id = ui.id "
        "WHERE ui.user_id = ? "
        "AND ts.is_trained = 1", 
        (session['user_id'],)).fetchall()

        trained_image_ids = [img[0] for img in trained_images]

        image_length = len(images)   
        trained_image_length = len(trained_images)

        return render_template(
                    "user-templates/profile.html",
                    images=images,
                    image_length=image_length,
                    trained_images=trained_image_ids,
                    trained_image_length=trained_image_length,
                )

@app.route('/addImages', methods=['POST'])
@login_required
def addImages():
    # Validate 'name' input
    name = request.form.get('name')
    if not name or name.strip() == '':
        flash('Name is required')
        return redirect(url_for('orgProfile'))
    
    if 'images' not in request.files:
        flash('No files part in the request')
        return redirect(url_for('orgProfile'))
    
    files = request.files.getlist('images')
    
    if len(files) == 0:
        flash('No files selected')
        return redirect(url_for('orgProfile'))

    if len(files) > 9:
        flash('You can only upload up to 9 images')
        return redirect(url_for('orgProfile'))
    
    user_id = session['user_id']
    db, c = get_db()

    # --- Update user's name in the users table ---
    c.execute('''
        UPDATE users
        SET user_name = ?
        WHERE id = ?
    ''', (name.strip(), user_id))

    session['user_name'] = name.strip()


    c.execute("SELECT COUNT(*) FROM userImages WHERE user_id = ?", (user_id,))
    existing_images_count = c.fetchone()[0]

    if existing_images_count >= 9:
        flash('You already have 9 images uploaded')
        return redirect(url_for('orgProfile'))

    uploaded_count = 0  # <-- Track successful uploads

    for file in files:
        if file and allowed_file(file.filename) and file.filename != '':
            if existing_images_count < 9:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                normalized_path = file_path.replace("\\", "/")
                
                c.execute('''
                    INSERT INTO userImages (user_id, image_path) 
                    VALUES (?, ?)
                ''', (user_id, normalized_path))
                
                existing_images_count += 1
                uploaded_count += 1
            else:
                flash('Upload limit reached. Some images were not uploaded.')
                break
        else:
            if file.filename != '':
                flash(f'Invalid file type: {file.filename}')

    db.commit()
    db.close()

    if uploaded_count > 0:
        flash('Images uploaded successfully')
    else:
        flash('No valid images were uploaded')

    return redirect(url_for('orgProfile'))


@app.route('/reset-user-images', methods=['POST'])
@login_required
def reset_user_images():
    db, cursor = get_db()
    user_id = session['user_id']

    # 🔍 Step 1: Get all image IDs & file paths for the logged-in user
    cursor.execute("SELECT id, image_path FROM userImages WHERE user_id = ?", (user_id,))
    user_images = cursor.fetchall()

    if not user_images:
        return "No images found to reset.", 200

    # 🗑 Step 2: Delete only the logged-in user's images from userImages
    cursor.execute("DELETE FROM userImages WHERE user_id = ?", (user_id,))
    db.commit()

    # 🗑 Step 3: Delete trainingStatus records related to those images
    image_ids = [str(img[0]) for img in user_images]
    cursor.execute("DELETE FROM trainingStatus WHERE image_id IN ({})".format(
        ",".join(["?"] * len(image_ids))
    ), image_ids)
    db.commit()

    # 🧹 Step 4: Delete only the logged-in user's uploaded images from 'static/uploads'
    for _, image_path in user_images:
        file_path = os.path.join(UPLOAD_FOLDER, image_path)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')

    flash("Your images and training data have been reset.", "success")
    return redirect(request.referrer)


@app.route('/trainImages', methods=['POST'])
@login_required
def trainImages():
    user_id = session['user_id']
    user_name = session['user_name']
    db, c = get_db()
    
    c.execute("SELECT id, image_path FROM userImages WHERE user_id = ?", (user_id,))
    user_images = c.fetchall()
    
    if len(user_images) < 9:
        flash('You need 9 images uploaded to start training')
        return redirect(url_for('orgProfile'))
    
    upload_folder_location = app.config['UPLOAD_FOLDER']
    trained_count = 0
    
    for image_id, image_filename in user_images:
        image_path = os.path.join(upload_folder_location, image_filename)
        success, message = train_image(image_filename, user_name, user_id)
        
        if success:
            trained_count += 1
            c.execute('''
                INSERT OR REPLACE INTO trainingStatus (image_id, is_trained, trained_at)
                VALUES (?, ?, ?)
            ''', (image_id, True, datetime.datetime.now()))
        else:
            flash(f'Error training image {image_filename}: {message}')
    
    if trained_count == len(user_images):
        flash('Training completed successfully')
    else:
        flash(f'Training completed with some errors. {trained_count}/{len(user_images)} trained successfully.')
    
    db.commit()
    db.close()
    return redirect(url_for('orgProfile'))


@app.route('/admin_clear_all_attendance')
def admin_clear_all_attendance():
    # Add admin check here
    db, cursor = get_db()
    cursor.execute("DELETE FROM attendance")
    db.commit()
    return "All attendance records cleared.", 200
