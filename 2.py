import streamlit as st
import sqlite3
import hashlib
import os
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import pandas as pd

# Utility Functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Database Functions
def register_user(username, password, role):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    try:
        c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, hashed_password, role))
        conn.commit()
        st.success("User registered successfully.")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ? AND password = ?', (username, password))
    user = c.fetchone()
    conn.close()
    return user

def get_all_users():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT id, username, role FROM users')
    users = c.fetchall()
    conn.close()
    return users

def add_image_to_db(filename, features):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO images (filename) VALUES (?)', (filename,))
    conn.commit()
    conn.close()
    
    # Load existing features and filenames
    Image_features = pkl.load(open('Images_features old.pkl', 'rb'))
    filenames = pkl.load(open('filenames old.pkl', 'rb'))
    
    # Append new features and filename
    Image_features = np.append(Image_features, [features], axis=0)
    filenames.append(filename)
    
    # Save updated features and filenames
    pkl.dump(Image_features, open('Images_features old.pkl', 'wb'))
    pkl.dump(filenames, open('filenames old.pkl', 'wb'))
    
    # Re-fit the Nearest Neighbors model with updated features
    neighbors.fit(Image_features)

def get_all_images():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT filename FROM images')
    images = c.fetchall()
    conn.close()
    return [image[0] for image in images]

def load_model():
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    model.trainable = False
    model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])
    return model

def extract_features_from_image(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

def initialize_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database
initialize_db()

# Load pre-trained model and data
model = load_model()
Image_features = pkl.load(open('Images_features old.pkl', 'rb'))
filenames = pkl.load(open('filenames old.pkl', 'rb'))

# Initialize Nearest Neighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Streamlit Interface
st.header('Fashion Recommendation System')

# Session State Management
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None

# Navigation
if st.session_state.authenticated:
    if st.session_state.role == 'admin':
        page = st.sidebar.selectbox("Select a page", ["User Details", "Image Upload", "Recommendation System"])
    else:
        page = st.sidebar.selectbox("Select a page", ["Recommendation System"])
    if st.sidebar.button('Logout'):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.role = None
        st.experimental_rerun()
else:
    page = st.sidebar.selectbox("Select a page", ["Login", "Register"])

# Registration Page
if page == "Register":
    st.header('User Registration')
    with st.form("registration_form"):
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        role = st.selectbox('Role', ['user'])
        submit_button = st.form_submit_button("Create Account")
        if submit_button:
            if username and password:
                register_user(username, password, role)
            else:
                st.error("Please provide both username and password.")

# Authentication Page
if page == "Login":
    if not st.session_state.authenticated:
        with st.form("login_form"):
            st.subheader('Login')
            username = st.text_input('Username')
            password = st.text_input('Password', type='password')
            submit_button = st.form_submit_button("Login")
            if submit_button:
                if username and password:
                    hashed_password = hash_password(password)
                    user = authenticate_user(username, hashed_password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.role = user[0]
                        page = "Recommendation System"  # Redirect to recommendation system
                        st.experimental_rerun()
                    else:
                        st.error('Invalid username or password')
    else:
        st.subheader(f'Welcome, {st.session_state.username}!')
        if st.button('Logout'):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.role = None
            st.experimental_rerun()

# Recommendation System Page
if page == "Recommendation System":
    if st.session_state.authenticated:
        st.subheader('Fashion Recommendation System')
        upload_file = st.file_uploader("Upload Image", key="recommendation_image_upload")
        if upload_file is not None:
            image_path = os.path.join('images1', upload_file.name)
            with open(image_path, 'wb') as f:
                f.write(upload_file.getbuffer())
            st.image(upload_file)
            
            input_img_features = extract_features_from_image(image_path, model)
            distance, indices = neighbors.kneighbors([input_img_features])
            st.subheader('Recommended Images')

            columns = st.columns(5)
            for i, idx in enumerate(indices[0][1:6]):
                # Correctly access filenames from the list
                full_image_path = filenames[idx]
                # full_image_path = os.path.join(' ', filename)
                # st.write(f"Displaying image {i+1} at path: {full_image_path}")  # Log the image path
                if os.path.exists(full_image_path):
                    try:
                        with columns[i]:
                            st.image(full_image_path)
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")
                        st.write(f"Filename causing issue: {full_image_path}")
                else:
                    st.error(f"Image not found: {full_image_path}")
    else:
        st.error("Please login to access the recommendation system.")

# User Details Page
if page == "User Details" and st.session_state.role == 'admin':
    st.subheader('User Details')
    users = get_all_users()
    user_df = pd.DataFrame(users, columns=['ID', 'Username', 'Role'])
    st.dataframe(user_df)

# Image Upload Page
if page == "Image Upload" and st.session_state.role == 'admin':
    st.subheader('Add Image to Dataset')
    upload_file_admin = st.file_uploader("Upload Image", key="admin_image_upload")
    if upload_file_admin is not None:
        image_path = os.path.join('images1', upload_file_admin.name)
        with open(image_path, 'wb') as f:
            f.write(upload_file_admin.getbuffer())
        st.image(upload_file_admin)
        
        # Extract features from the new image
        features = extract_features_from_image(image_path, model)
        
        # Add image and its features to the database and update PKL files
        add_image_to_db(upload_file_admin.name, features)
        
        # Re-fit the Nearest Neighbors model with updated features
        Image_features = pkl.load(open('Images_features old.pkl', 'rb'))
        filenames = pkl.load(open('filenames old.pkl', 'rb'))
        neighbors.fit(Image_features)
        
        st.success('Image added to dataset and features updated')
