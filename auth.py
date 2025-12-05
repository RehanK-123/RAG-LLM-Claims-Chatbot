import time
import bcrypt
from auth_db import get_connection

LOCK_THRESHOLD = 5        # attempts before lockout
LOCK_TIME_SECONDS = 120   # 2 minutes

def create_user(username: str, password: str):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    
    try:
        conn = get_connection()
        conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                     (username, hashed.decode()))
        conn.commit()
        return True, "Account created successfully."
    except Exception as e:
        return False, "Username already exists."


def authenticate(username: str, password: str):
    conn = get_connection()
    user = conn.execute(
        "SELECT id, password_hash, failed_attempts, locked_until FROM users WHERE username=?",
        (username,)
    ).fetchone()

    if not user:
        return False, "Invalid username or password."

    user_id, stored_hash, attempts, locked_until = user

    # Check lock status
    if locked_until and time.time() < locked_until:
        return False, f"Account locked. Try again in {int(locked_until - time.time())}s."

    # Validate password
    if bcrypt.checkpw(password.encode(), stored_hash.encode()):
        conn.execute("UPDATE users SET failed_attempts=0, locked_until=NULL WHERE id=?", (user_id,))
        conn.commit()
        return True, "Authenticated successfully."

    # Wrong password → increment and maybe lock
    attempts += 1
    lock_val = None

    if attempts >= LOCK_THRESHOLD:
        lock_val = time.time() + LOCK_TIME_SECONDS

    conn.execute("UPDATE users SET failed_attempts=?, locked_until=? WHERE id=?", 
                 (attempts, lock_val, user_id))
    conn.commit()

    if lock_val:
        return False, f"Account locked for {LOCK_TIME_SECONDS//60} minutes."

    return False, f"Incorrect password. {LOCK_THRESHOLD - attempts} attempts left."
