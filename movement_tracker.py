import sqlite3
from collections import defaultdict
import time

# Track movement frames
movement_tracker = {}

# Track face movements
face_tracks = defaultdict(list)

def track_movement(name, x, y, get_db):
    """Tracks the movement of a face and confirms entry/exit."""
    print(f"track_movement called for {name} at ({x}, {y})")  # Debugging print
    
    user_id = name.split("_")[0]  # Extract user ID
    print(f"Extracted user_id: {user_id}")
    
    if user_id not in movement_tracker:
        movement_tracker[user_id] = {"frames": 0, "status": None, "last_db_status": None}
        print(f"Initializing movement tracking for user {user_id}")
    
    movement_tracker[user_id]["frames"] += 1
    print(f"User {user_id} frame count: {movement_tracker[user_id]['frames']}")
    
    face_tracks[user_id].append((x, y))
    print(f"Updated face_tracks[{user_id}]: {face_tracks[user_id]}")
    
    if len(face_tracks[user_id]) > 2:
        prev_x, _ = face_tracks[user_id][-2]
        curr_x, _ = face_tracks[user_id][-1]
        
        print(f"Previous x: {prev_x}, Current x: {curr_x}")
        
        if prev_x < curr_x - 10:
            status = "entered"
        elif prev_x > curr_x + 10:
            status = "exited"
        else:
            status = None
        
        print(f"Detected movement status: {status}")
        
        if status is not None:
            print(f"{name} = {status}")
            movement_tracker[user_id]["status"] = status
            
            # Ensure 25-frame consistency before marking attendance
            if movement_tracker[user_id]["frames"] >= 25:
                print(f"User {user_id} has been consistently detected for 25 frames, updating attendance.")
                update_attendance(user_id, status, get_db)

def update_attendance(user_id, status, get_db):
    """Updates the attendance database if the user state has changed."""
    print(f"update_attendance called for user {user_id} with status {status}")  # Debugging print
    
    db, c = get_db()
    
    # Check last recorded status
    c.execute("SELECT status FROM attendance WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1", (user_id,))
    last_status = c.fetchone()
    print(f"Last recorded status for user {user_id}: {last_status}")
    
    if last_status and last_status[0] == status:
        print(f"User {user_id} already marked as {status}, skipping duplicate entry.")
        return  # Avoid duplicate entry
    
    # Insert attendance record
    c.execute("INSERT INTO attendance (user_id, status) VALUES (?, ?)", (user_id, status))
    db.commit()
    
    print(f"User {user_id} marked as {status} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Reset frame count for user
    movement_tracker[user_id]["frames"] = 0
    print(f"Resetting frame count for user {user_id}")
