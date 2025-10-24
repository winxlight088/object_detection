from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import sys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

app.secret_key = 'your_secret_key_here'  # Change this to a random secret key in production
USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.json')

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('signin'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users:
            flash('Username already exists!')
            return redirect(url_for('signup'))
        users[username] = generate_password_hash(password)
        save_users(users)
        flash('Sign up successful! Please sign in.')
        return redirect(url_for('signin'))
    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            flash('Signed in successfully!')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password!')
            return redirect(url_for('signin'))
    return render_template('signin.html')

@app.route('/signout')
def signout():
    session.pop('username', None)
    flash('Signed out successfully!')
    return redirect(url_for('signin'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/car')
@login_required
def car():
    return render_template('car.html')

@app.route('/people')
@login_required
def people():
    return render_template('people.html')

@app.route('/construction')
@login_required
def construction():
    return render_template('construction.html')

@app.route('/poker')
@login_required
def poker():
    return render_template('poker.html')

@app.route('/detect/car', methods=['POST'])
@login_required
def detect_car():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'car_' + file.filename)
    # Run car_counter.py as a subprocess
    import subprocess
    import shutil
    try:
        # Use sys.executable to ensure the same Python interpreter is used
        result = subprocess.run([
            sys.executable,
            'project-1_car-counter/car_counter.py',
            '--input', filepath,
            '--output', processed_path
        ], capture_output=True, text=True, check=True, cwd=os.getcwd())
        # Optionally parse stdout for car count
        car_count = 0
        for line in result.stdout.splitlines():
            if 'Total vehicles counted:' in line:
                try:
                    car_count = int(line.split(':')[-1].strip())
                except Exception:
                    pass
        status = 'Detection Complete'
    except subprocess.CalledProcessError as e:
        car_count = 0
        status = f'Error: {e.stderr}'
        # fallback: copy input to output
        shutil.copy(filepath, processed_path)
    return jsonify({'output_url': '/' + processed_path, 'count': car_count, 'status': status})

@app.route('/detect/people', methods=['POST'])
@login_required
def detect_people():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'people_' + file.filename)
    import subprocess
    import shutil
    try:
        result = subprocess.run([
            sys.executable,
            'project-2_people-counter/people_counter.py',
            '--input', filepath,
            '--output', processed_path
        ], capture_output=True, text=True, check=True, cwd=os.getcwd())
        people_count = 0
        for line in result.stdout.splitlines():
            if 'People Count:' in line:
                try:
                    people_count = int(line.split(':')[-1].strip())
                except Exception:
                    pass
        status = 'Detection Complete'
    except subprocess.CalledProcessError as e:
        people_count = 0
        status = f'Error: {e.stderr}'
        shutil.copy(filepath, processed_path)
    return jsonify({'output_url': '/' + processed_path, 'count': people_count, 'status': status})

@app.route('/detect/construction', methods=['POST'])
@login_required
def detect_construction():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'ppe_' + file.filename)
    import subprocess
    import shutil
    try:
        result = subprocess.run([
            sys.executable,
            'project-3_construction/ppe_detection.py',
            '--input', filepath,
            '--output', processed_path
        ], capture_output=True, text=True, check=True, cwd=os.getcwd())
        status = 'Detection Complete'
    except subprocess.CalledProcessError as e:
        status = f'Error: {e.stderr}'
        shutil.copy(filepath, processed_path)
    return jsonify({'output_url': '/' + processed_path, 'status': status})

@app.route('/detect/poker', methods=['POST'])
@login_required
def detect_poker():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'poker_' + file.filename)
    import subprocess
    import re
    import shutil
    try:
        result = subprocess.run([
            sys.executable,
            'project-4_poker_card_detection/PokerHandDetector.py',
            '--input', filepath,
            '--output', processed_path
        ], capture_output=True, text=True, check=True, cwd=os.getcwd())
        # Try to extract card names from stdout
        cards = []
        for line in result.stdout.splitlines():
            if line.startswith('Cards:'):
                cards = [c.strip() for c in line.split(':', 1)[-1].split(',') if c.strip()]
        status = 'Detection Complete'
    except subprocess.CalledProcessError as e:
        cards = []
        status = f'Error: {e.stderr}'
        shutil.copy(filepath, processed_path)
    return jsonify({'output_url': '/' + processed_path, 'cards': cards, 'status': status})

@app.route('/detect/poker/webcam', methods=['POST'])
@login_required
def detect_poker_webcam():
    import numpy as np
    import cv2
    import math
    import sys
    import io
    import cvzone
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../project-4_poker_card_detection')))
    import importlib.util
    poker_func_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../project-4_poker_card_detection/PokerHandFunction.py'))
    spec = importlib.util.spec_from_file_location('PokerHandFunction', poker_func_path)
    PokerHandFunction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PokerHandFunction)
    from ultralytics import YOLO
    # Get the image from the POST request
    file = request.files['frame']
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'project-4_poker_card_detection', 'playingCards.pt')
    model = YOLO(model_path)
    classNames = ['10C', '10D', '10H', '10S',
                  '2C', '2D', '2H', '2S',
                  '3C', '3D', '3H', '3S',
                  '4C', '4D', '4H', '4S',
                  '5C', '5D', '5H', '5S',
                  '6C', '6D', '6H', '6S',
                  '7C', '7D', '7H', '7S',
                  '8C', '8D', '8H', '8S',
                  '9C', '9D', '9H', '9S',
                  'AC', 'AD', 'AH', 'AS',
                  'JC', 'JD', 'JH', 'JS',
                  'KC', 'KD', 'KH', 'KS',
                  'QC', 'QD', 'QH', 'QS']
    results = model(img, stream=True)
    hand = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            if conf > 0.5:
                hand.append(classNames[cls])
    hand = list(set(hand))
    hand_name = ""
    if len(hand) == 5:
        hand_name = PokerHandFunction.findPokerHand(hand)
        cvzone.putTextRect(img, f'Your Hand: {hand_name}', (30, 75), scale=2, thickness=4)
    # Encode the processed image as JPEG
    _, jpeg = cv2.imencode('.jpg', img)
    return (jpeg.tobytes(), 200, {'Content-Type': 'image/jpeg', 'X-Hand-Name': hand_name})

if __name__ == '__main__':
    app.run(debug=True)
