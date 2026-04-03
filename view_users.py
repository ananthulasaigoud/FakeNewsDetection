import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "users.db")

def view_users():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at: {DB_PATH}")
        print("You haven't signed up any users yet!")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT id, username, contact, email, address FROM users")
        users = cursor.fetchall()

        if not users:
            print("No users found in the database. (The table is empty)")
        else:
            print(f"--- Found {len(users)} User(s) ---")
            for u in users:
                print(f"Username : {u['username']}")
                print(f"Contact  : {u['contact'] or 'None'}")
                print(f"Email    : {u['email'] or 'None'}")
                print(f"Address  : {u['address'] or 'None'}")
                print("-" * 30)
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    view_users()
