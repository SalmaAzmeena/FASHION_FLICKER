import sqlite3
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def insert_test_data():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Insert test users
    users = [
        
        ('admin', hash_password('admin123'), 'admin')
    ]
    
    c.executemany('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', users)
    
    # Commit changes
    conn.commit()
    conn.close()
    print("Test data inserted.")

if __name__ == "__main__":
    insert_test_data()
