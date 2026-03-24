from bson import ObjectId
from flask import Flask, request, render_template, redirect, url_for, flash, session , Response , jsonify
import os
import cv2
import numpy as np
from datetime import date, datetime, timedelta
import joblib
from sklearn.neighbors import KNeighborsClassifier
from gtts import gTTS
from io import BytesIO
import pygame
import smtplib
from email.mime.text import MIMEText
from cryptography.fernet import Fernet
import logging
from pymongo import MongoClient
import base64
import csv
from io import StringIO
import gspread
from google.oauth2.service_account import Credentials
from gspread_formatting import *
import smtplib
from email.mime.text import MIMEText
import logging
import requests
from datetime import datetime
import pytz
import gspread
from google.oauth2.service_account import Credentials
from io import StringIO
import csv
from flask import Response
from gspread_formatting import get_conditional_format_rules, ConditionalFormatRule, GridRange, BooleanRule, BooleanCondition, CellFormat, Color, TextFormat
import gspread
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from googleapiclient.errors import HttpError
from mailtm import Email

#### Defining Flask App
app = Flask(__name__)
app.secret_key = os.urandom(24)

#### Logging Setup
logging.basicConfig(filename='app.log', level=logging.INFO)

#### Initialize encryption (optional)
key = Fernet.generate_key()
cipher_suite = Fernet(key)

#### MongoDB Setup
client = MongoClient('mongodb://localhost:27017/')
db = client.attendance_db
users_collection = db.users
grace_attendance_collection = db.grace_attendance
grace_request_collection = db.grace_request
attendance_collection = db.attendance
time_periods_collection = db.time_periods
feedback_collection = db.feedback


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

user_roles_collection = db.user_roles

# Helper to check permissions in templates
@app.context_processor
def utility_processor():
    def get_user_role():
        if 'email' in session:
            user_data = user_roles_collection.find_one({"email": session['email']})
            return user_data.get('role', 'student') if user_data else 'student'
        return None
    return dict(get_user_role=get_user_role)


@app.route('/set_session', methods=['POST'])
def set_session():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')

    if not email:
        return jsonify({'status': 'failed', 'message': 'Email is required'}), 400

    # Check for role in the user_roles collection
    user_info = user_roles_collection.find_one({"email": email})
    
    if not user_info:
        # If user is new, default to 'student' role
        user_roles_collection.insert_one({
            "email": email,
            "role": "student",
            "name": username
        })
        role = "student"
    else:
        role = user_info.get('role', 'student')

    session['logged_in'] = True
    session['username'] = username
    session['email'] = email
    session['role'] = role 
    
    return jsonify({'status': 'success', 'role': role}), 200

#### Utility Functions
def totalreg():
    return users_collection.count_documents({})

def extract_faces(img):
    try:
        if img.shape != (0, 0, 0):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_points = face_detector.detectMultiScale(gray, 1.3, 5)
            return face_points
        else:
            return []
    except Exception as e:
        logging.error(f"Error in extract_faces: {e}")
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = users_collection.find()
    for user in userlist:
        for face_data in user['faces']:
            img_data = base64.b64decode(face_data)
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(f"{user['name']}_{user['id']}")
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    attendance_records = attendance_collection.find({"date": datetoday})
    names, rolls, times = [], [], []
    for record in attendance_records:
        names.append(record['name'])
        rolls.append(record['roll'])
        times.append(record['time'])
    l = len(names)
    return names, rolls, times, l

def getallusers():
    userlist = users_collection.find()
    names, rolls = [], []
    for user in userlist:
        names.append(user['name'])
        rolls.append(user['id'])
    l = len(names)
    return userlist, names, rolls, l

def play_voice_message(message):
    tts = gTTS(text=message, lang='en')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)

    pygame.mixer.init()
    pygame.mixer.music.load(fp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def add_attendance(name):

    # 1) Get current session
    current_period = get_time_period()

    # Ignore silently if outside any session
    if current_period is None:
        return False


    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")

    # 2) Check duplicate ONLY FOR SAME SESSION
    exists = attendance_collection.find_one({
        "roll": userid,
        "date": datetoday,
        "session": current_period      
    })

    if exists:
        return False


    # 3) Insert session wise attendance
    attendance_collection.insert_one({
        "name": username,
        "roll": userid,
        "time": current_time,
        "date": datetoday,

        # 🔥 NEW COLUMN
        "session": current_period
    })

    return True


#### Route Functions
@app.route('/')
def home():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    names, rolls, times, l = extract_attendance()
    email_configured = db.email_settings.find_one({}) is not None
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, email_configured=email_configured)  

@app.route('/start', methods=['GET'])
def start():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, times, l = extract_attendance()
        return render_template(
            'home.html',
            names=names, rolls=rolls, times=times, l=l,
            totalreg=totalreg(),
            datetoday2=datetoday2,
            email_configured=db.email_settings.find_one({}) is not None,
            mess='There is no trained model in the static folder. Please add a new face to continue.'
        )

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            
            # Split and query the database
            username, userid = identified_person.split('_')
         
            # print(f"Querying for: name={username}, id={userid}")
            
            user = users_collection.find_one({"name": username, "id": userid})
            # print(f"User found: {user}")
            
            if add_attendance(identified_person):
                if user:
                    student_email = user.get('email')
                    if student_email:
                        email_context = {
                            'name': username,
                            'id': userid,
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'time': datetime.now().strftime('%H:%M:%S')
                        }
                        send_email(
                            'BIT Attendance Update',
                            email_context,
                            student_email
                        )
                    else:
                        logging.warning(f"No email for user {username}_{userid}; attendance email not sent.")
                play_voice_message('Attendance marked successfully.')
            cv2.putText(
                frame, 
                f'{identified_person}', 
                (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 0, 20), 
                2, 
                cv2.LINE_AA
            )
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template(
        'home.html',
        names=names,
        rolls=rolls,
        times=times,
        l=l,
        totalreg=totalreg(),
        datetoday2=datetoday2,
        email_configured=db.email_settings.find_one({}) is not None
    )

@app.route('/email-settings', methods=['GET', 'POST'])
def email_settings():
    if request.method == 'POST':
        sender_email = request.form.get('sender_email', '').strip()
        sender_password = request.form.get('sender_password', '')

        # Validate the email address using ZeroBounce API (skip if API fails so user can still save)
        if sender_email and not validate_email(sender_email):
            email_settings = db.email_settings.find_one({})
            return render_template(
                'email_settings.html',
                email_settings={'sender_email': sender_email, 'sender_password': sender_password or (email_settings.get('sender_password') if email_settings else '')},
                mess="Invalid email address. Please enter a valid sender email."
            )

        # Check if email settings already exist
        existing_settings = db.email_settings.find_one({})
        if existing_settings:
            # Update: keep existing password if user left it blank
            update_data = {"sender_email": sender_email}
            if sender_password:
                update_data["sender_password"] = sender_password
            db.email_settings.update_one(
                {"_id": existing_settings['_id']},
                {"$set": update_data}
            )
        else:
            # Insert new settings (password required for first-time setup)
            if not sender_password:
                return render_template(
                    'email_settings.html',
                    email_settings={'sender_email': sender_email, 'sender_password': ''},
                    mess="Please enter sender password (use Gmail App Password for Gmail)."
                )
            db.email_settings.insert_one({"sender_email": sender_email, "sender_password": sender_password})

        flash("Email settings updated successfully.", "success")
        return redirect(url_for('email_settings'))

    # Fetch existing settings for display
    email_settings = db.email_settings.find_one({})
    return render_template('email_settings.html', email_settings=email_settings)


def send_email(subject, body_context, to_email):
    try:
        # Fetch email settings from the database
        email_settings = db.email_settings.find_one({})
        if not email_settings:
            logging.error("Email settings are missing in the database. Configure sender email at /email-settings.")
            return False

        sender_email = email_settings['sender_email']
        sender_password = email_settings['sender_password']

        if not sender_email or not sender_password:
            logging.error("Sender email or password is empty in email settings.")
            return False

        # Render the HTML template with context
        html_body = render_template('email_template.html', **body_context)

        # Create the email message
        msg = MIMEText(html_body, 'html')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = to_email

        # Connect to the SMTP server and send
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [to_email], msg.as_string())
        logging.info(f"Email successfully sent to {to_email}")
        return True
    except smtplib.SMTPAuthenticationError as e:
        logging.error(f"SMTP authentication failed. For Gmail, use an App Password: {e}")
        return False
    except Exception as e:
        logging.error(f"Error in sending email: {e}")
        return False



def validate_email(email):
    api_key = 'fd9b5c9bf5c14840930b8a87c96e454f'  # Your ZeroBounce API key
    url = f'https://api.zerobounce.net/v2/validate?api_key={api_key}&email={email}'
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'valid':
            return True
        else:
            return False
    else:
        logging.error(f"Error validating email: {response.status_code}")
        return False


@app.route('/add', methods=['GET', 'POST'])
def add():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        newuseremail = request.form['newuseremail']
        # New Fields captured from form
        newphone = request.form['newphone']
        newdept = request.form['newdept']
        newyear = request.form['newyear']
        newsemester = request.form['newsemester']
        
        if not validate_email(newuseremail):
            return render_template('add.html', mess="Invalid email address.")
        
        user_faces = []
        i, j = 0, 0
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                if j % 10 == 0:
                    face = frame[y:y+h, x:x+w]
                    resized_face = cv2.resize(face, (50, 50))
                    encoded_face = base64.b64encode(cv2.imencode('.jpg', resized_face)[1]).decode()
                    user_faces.append(encoded_face)
                    i += 1
                j += 1
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27 or i == 20:
                break
        cap.release()
        cv2.destroyAllWindows()
        
        users_collection.insert_one({
            "name": newusername, 
            "id": newuserid, 
            "email": newuseremail,
            "phone": newphone,
            "department": newdept,
            "year": newyear,
            "semester": newsemester,
            "faces": user_faces
        })
        
        logging.info(f'User {newusername} added with full details.')
        train_model()
        return redirect(url_for('home'))
    return render_template('add.html')

@app.route('/edit_user/<username>', methods=['GET', 'POST'])
def edit_user(username):
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    # Splitting the username into name and id components to locate the user
    name_part, id_part = username.split('_')
    user = users_collection.find_one({'name': name_part, 'id': id_part})

    if not user:
        return 'User not found!', 404

    if request.method == 'POST':
        new_username = request.form['newname']  # Updated to match HTML form field
        new_userid = request.form['newroll']   # Updated to match HTML form field
        new_useremail = request.form['newemail']  # New email field

                # Validate the email address using ZeroBounce API
        if not validate_email(new_useremail):
            return render_template(
                'add.html',
                user=user,
                mess="Invalid email address. Please enter a valid email."
            )

        # Update the user document with the new name and ID
        users_collection.update_one(
            {'name': name_part, 'id': id_part}, 
            {'$set': {'name': new_username, 'id': new_userid,'email': new_useremail}}
        )

        logging.info('User information updated and model retrained')
        train_model()  # Retrain model after updating user data

        return redirect(url_for('home'))
    
    return render_template('edit_user.html', user=user)

@app.route('/update_student_details', methods=['GET', 'POST'])
def update_student_details():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_id = request.form['user_id']
        
        # Update with new attributes
        users_collection.update_one(
            {'id': user_id},
            {'$set': {
                'name': request.form['new_name'],
                'id': request.form['new_roll'],
                'email': request.form['new_email'],
                'phone': request.form['new_phone'],
                'department': request.form['new_dept'],
                'year': request.form['new_year'],
                'semester': request.form['new_semester']
            }}
        )

        logging.info(f'Student details for ID {user_id} updated.')
        train_model()
        return redirect(url_for('update_student_details'))

    # Fetch all users and ensure the dictionary includes all new keys
    users = users_collection.find()
    formatted_users = []
    for user in users:
        formatted_users.append({
            'name': user.get('name'),
            'id': user.get('id'),
            'email': user.get('email'),
            'phone': user.get('phone', 'N/A'),
            'department': user.get('department', 'N/A'),
            'year': user.get('year', 'N/A'),
            'semester': user.get('semester', 'N/A'),
            'face_image': user['faces'][0] if 'faces' in user else None
        })

    return render_template('update_student_details.html', users=formatted_users)

@app.route('/delete_student/<user_id>', methods=['POST'])
def delete_student(user_id):
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    # Delete user from the 'users' collection
    users_collection.delete_one({'id': user_id})

    # Delete attendance records for the user
    attendance_collection.delete_many({'roll': user_id})

    logging.info(f"Deleted student with ID {user_id} and their attendance records.")
    
    return redirect(url_for('update_student_details'))

# Function to load time periods from the database
def load_time_periods():
    time_periods = time_periods_collection.find_one({"_id": 1})
    if not time_periods:
        return {}  # Return an empty dictionary if no time periods are set
    return {
        "forenoon": {"start": time_periods["foornoon"]["start"], "end": time_periods["foornoon"]["end"], "sheet_name": "sheet1"},
        "afternoon": {"start": time_periods["afternoon"]["start"], "end": time_periods["afternoon"]["end"], "sheet_name": "sheet2"},
        "night": {"start": time_periods["night"]["start"], "end": time_periods["night"]["end"], "sheet_name": "sheet3"}
    }

# Route for setting time periods
@app.route('/set_time_periods', methods=['GET', 'POST'])
def set_time_periods():
    if request.method == 'POST':
        # Get values from the form
        forenoon_start = request.form.get('foornoon_start')
        forenoon_end = request.form.get('foornoon_end')
        afternoon_start = request.form.get('afternoon_start')
        afternoon_end = request.form.get('afternoon_end')
        night_start = request.form.get('night_start')
        night_end = request.form.get('night_end')

        # Update the time periods in the MongoDB collection
        time_periods_collection.update_one(
            {"_id": 1},  # Assuming there’s a document with _id = 1 that holds the time periods
            {
                "$set": {
                    "foornoon": {"start": forenoon_start, "end": forenoon_end},
                    "afternoon": {"start": afternoon_start, "end": afternoon_end},
                    "night": {"start": night_start, "end": night_end}
                }
            },
            upsert=True  # If the document doesn't exist, it will be created
        )

        return redirect(url_for('set_time_periods'))  # Redirect to refresh the page

    # Fetch current time periods from MongoDB
    time_periods = time_periods_collection.find_one({"_id": 1})

    return render_template('set_time_periods.html', time_periods=time_periods)

# Function to get the current time period
def get_time_period():
    current_time = datetime.now(pytz.timezone("Asia/Kolkata")).strftime('%H:%M')
    time_periods = load_time_periods()  # Load time periods dynamically
    for period, times in time_periods.items():
        if times["start"] <= current_time <= times["end"]:
            return period
    return None  # Outside defined time periods

@app.route('/attendance_log')
def attendance_log():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    # Fetch all attendance records from the database
    attendance_records = attendance_collection.find().sort("date", -1)
    
    # Structure data for easy display
    attendance_data = []
    for record in attendance_records:
        user = users_collection.find_one({'id': record['roll']})
        face_image = user['faces'][0] if user and 'faces' in user else None
        status = "Present" if record else "Absent"

        attendance_data.append({
            "name": record['name'],
            "roll": record['roll'],
            "time": record['time'],
            "date": record['date'],
            "face_image": face_image,
            "status": status
        })
    
    # Update Google Sheet with time-period-specific data
    time_period = get_time_period()
    if time_period:
        update_google_sheet(attendance_data, time_period)
    
    return render_template('attendance_log.html', attendance_data=attendance_data)

def update_google_sheet(attendance_data, time_period):
    try:
        # Path to your service account JSON file
        credentials_path = os.getenv("SERVICE_ACCOUNT_PATH")
        if not credentials_path:
            raise ValueError("SERVICE_ACCOUNT_PATH environment variable is missing.")
        
        # Define the scope and authenticate
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials = Credentials.from_service_account_file(credentials_path, scopes=scopes)
        
        # Check if credentials need to be refreshed
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        
        # Authorize with gspread
        client = gspread.authorize(credentials)
        
        # Open the Google Sheet by URL
        sheet_url = os.getenv("GOOGLE_SHEET_URL")
        if not sheet_url:
            raise ValueError("GOOGLE_SHEET_URL environment variable is missing.")
        
        sheet = client.open_by_url(sheet_url)
        
        # Select the correct sheet based on the time period
        time_periods = load_time_periods()
        sheet_name = time_periods[time_period]["sheet_name"]
        
        # Try to get the worksheet by name (sheet1, sheet2, sheet3)
        try:
            sheet_instance = sheet.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound:
            raise ValueError(f"Sheet {sheet_name} not found in the Google Sheet.")
        
        # Clear existing data (optional)
        sheet_instance.clear()
        
        # Add headers (S No, Name, ID, Date, Time, Status)
        headers = ["S No", "Name", "ID", "Date", "Time", "Status"]
        sheet_instance.append_row(headers)

        # Add attendance data row by row
        for index, record in enumerate(attendance_data, start=1):
            sheet_instance.append_row([
                index,
                record['name'],
                record['roll'],
                record['date'],
                record['time'],
                record['status']
            ])
        
        # Get the range for the "Status" column (F2:F)
        status_range = f"F2:F{len(attendance_data) + 1}"
        
        # Define formatting rules (green for Present, red for Absent)
        rule_present = gspread.formatting.ConditionalFormatRule(
            ranges=[gspread.models.GridRange.from_a1_range(status_range, sheet_instance)],
            booleanRule=gspread.formatting.BooleanRule(
                condition=gspread.formatting.BooleanCondition("TEXT_EQ", ["Present"]),
                format=gspread.formatting.CellFormat(
                    backgroundColor=gspread.formatting.Color(0.85, 0.94, 0.84),  # Light green
                    textFormat=gspread.formatting.TextFormat(bold=True)
                ),
            ),
        )
        
        rule_absent = gspread.formatting.ConditionalFormatRule(
            ranges=[gspread.models.GridRange.from_a1_range(status_range, sheet_instance)],
            booleanRule=gspread.formatting.BooleanRule(
                condition=gspread.formatting.BooleanCondition("TEXT_EQ", ["Absent"]),
                format=gspread.formatting.CellFormat(
                    backgroundColor=gspread.formatting.Color(0.96, 0.8, 0.8),  # Light red
                    textFormat=gspread.formatting.TextFormat(bold=True)
                ),
            ),
        )
        
        # Apply the formatting rules
        rules = sheet_instance.get_conditional_format_rules()
        rules.clear()  # Clear existing rules
        rules.append(rule_present)
        rules.append(rule_absent)
        sheet_instance.save_conditional_format_rules(rules)

        print(f"Google Sheet updated successfully for {time_period} period.")

    except ValueError as ve:
        print(f"Error: {ve}")
    except gspread.exceptions.APIError as api_error:
        print(f"Google Sheets API error: {api_error}")
    except Exception as e:
        print(f"Unexpected error: {e}")

@app.route('/download_attendance_csv')
def download_attendance_csv():
    attendance_records = attendance_collection.find().sort("date", -1)
    output = StringIO()
    writer = csv.writer(output)
    
    # Updated Headers to include new fields
    writer.writerow(['S No', 'Name', 'ID', 'Dept', 'Year', 'Sem', 'Phone', 'Date', 'Time', 'Status']) 
    
    for index, record in enumerate(attendance_records, start=1):
        # Fetch extra details from the users collection using the roll number
        user = users_collection.find_one({'id': record['roll']})
        
        dept = user.get('department', 'N/A') if user else 'N/A'
        year = user.get('year', 'N/A') if user else 'N/A'
        sem = user.get('semester', 'N/A') if user else 'N/A'
        phone = user.get('phone', 'N/A') if user else 'N/A'

        writer.writerow([
            index, record['name'], record['roll'], 
            dept, year, sem, phone,
            record['date'], record['time'], "Present"
        ])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=detailed_attendance.csv"}
    )

@app.route('/delete_user/<username>', methods=['POST'])
def delete_user(username):
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    # Extract name and ID from the username parameter
    name, user_id = username.split('_')
    user_id = user_id

    # Delete attendance records for this user
    attendance_collection.delete_many({'roll': user_id})

    logging.info(f"Deleted attendance records for user {name} with ID {user_id}")
    
    return redirect(url_for('home'))

faculties_collection = db.faculties
user_roles_collection = db.user_roles

@app.route('/add_faculty', methods=['GET', 'POST'])
def add_faculty():
    if 'logged_in' not in session or session.get('role') != 'admin':
        flash("Unauthorized Access!", "danger")
        return redirect(url_for('home'))

    if request.method == 'POST':
        # Use .strip() to remove accidental spaces at start/end
        f_id = request.form.get('faculty_id').strip()
        name = request.form.get('name').strip()
        contact = request.form.get('contact').strip()
        email = request.form.get('email').strip().lower() # Standardize email to lowercase
        role = request.form.get('role')

        # 1. Check if Faculty ID or Email already exists in either collection
        existing_faculty = faculties_collection.find_one({"$or": [{"email": email}, {"id": f_id}]})
        existing_role = user_roles_collection.find_one({"email": email})

        if existing_faculty or existing_role:
            flash("Error: Faculty ID or Email is already registered in the system.", "warning")
        else:
            try:
                # 2. Insert into Faculties Collection (Full Details)
                faculties_collection.insert_one({
                    "id": f_id,
                    "name": name,
                    "contact": contact,
                    "email": email,
                    "role": role,
                    "date_added": datetime.now()
                })
                
                # 3. Insert into user_roles (Login Credentials)
                # Using insert_one instead of update_one to prevent overwriting logic errors
                user_roles_collection.insert_one({
                    "email": email,
                    "role": role,
                    "id": f_id
                })
                
                flash(f"Success! {name} registered as {role.capitalize()}.", "success")
                return redirect(url_for('add_faculty'))
            except Exception as e:
                flash(f"Database Error: {str(e)}", "danger")

    return render_template('add_faculty.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'bitsathy' and password == '1234':
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    return render_template('login.html')    

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/hostel_dashboard', methods=['GET', 'POST'])
def hostel_dashboard():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    user_role = session.get('role')
    user_email = session.get('email')

    # ===== DATE HANDLING =====
    selected_date = request.args.get("date")
    try:
        if selected_date:
            date_obj = datetime.strptime(selected_date, "%Y-%m-%d")
        else:
            date_obj = datetime.today()
    except:
        date_obj = datetime.today()

    db_date = date_obj.strftime("%m_%d_%y")
    selected_date = date_obj.strftime("%Y-%m-%d")

    # ===== GET HOSTEL STUDENTS (Master List) =====
    hostel_students = list(users_collection.find({
        "residence": {"$regex": "hostel", "$options": "i"}
    }))
    total_hostel = len(hostel_students)

    # ===== TODAY NIGHT SESSION STATS =====
    today_night = list(attendance_collection.find({
        "date": db_date,
        "session": "night"
    }))

    # Set of rolls present tonight (converted to string and stripped for safety)
    present_rolls = set([str(r['roll']).strip() for r in today_night])

    # --- NEW LOGIC TO BUILD LISTS ---
    present_list = []
    absent_list = []

    for student in hostel_students:
        # Match based on the student 'id' field
        student_id = str(student.get('id')).strip()
        
        if student_id in present_rolls:
            present_list.append(student)
        else:
            absent_list.append(student)

    # Pass the actual lists into the stats dictionary
    stats = {
        "total": total_hostel,
        "present_count": len(present_list),
        "absent_count": len(absent_list),
        "percent": round((len(present_list) / total_hostel) * 100, 2) if total_hostel else 0,
        "present_list": present_list,
        "absent_list": absent_list
    }

    # ===== INDIVIDUAL SEARCH =====
    search_data = None
    search_roll = None

    if user_role == 'student':
        student_rec = users_collection.find_one({"email": user_email})
        if student_rec:
            search_roll = str(student_rec.get('id'))
    elif request.method == 'POST':
        search_roll = request.form.get('search_roll')

    if search_roll:
        student = users_collection.find_one({
            "id": search_roll,
            "residence": {"$regex": "hostel", "$options": "i"}
        })

        if student:
            # Re-using student_id for clarity
            sid = str(student['id'])
            
            # Get all historical night records for this student
            history = list(attendance_collection.find({"roll": sid, "session": "night"}))
            attended_dates = {r['date'].strip() for r in history}

            # Find the very first attendance record in the DB to calculate start date
            first_rec = attendance_collection.find_one(sort=[("date", 1)])
            start = datetime.strptime(first_rec['date'], "%m_%d_%y") if first_rec else datetime.today()
            end = date_obj

            total_days = 0
            absent_display = []
            curr = start
            while curr <= end:
                d_db = curr.strftime("%m_%d_%y")
                total_days += 1
                if d_db not in attended_dates:
                    absent_display.append(curr.strftime("%d-%m-%Y"))
                curr += timedelta(days=1)

            search_data = {
                "student": student,
                "absent_dates": sorted(absent_display, reverse=True),
                "percentage": round((len(attended_dates) / total_days) * 100, 2) if total_days else 0,
                "today_status": "Present" if db_date in attended_dates else "Absent"
            }

    return render_template(
        'hostel_dashboard.html',
        stats=stats,
        search_data=search_data,
        selected_date=selected_date
    )

def get_working_days(start, end, holidays, weekly_off):

    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")

    weekly_off = [w.lower() for w in weekly_off]
    holidays = [h.strip() for h in holidays]

    total = 0
    current = s

    while current <= e:

        day_name = current.strftime("%A").lower()
        date_str = current.strftime("%Y-%m-%d")

        if day_name not in weekly_off and date_str not in holidays:
            total += 1

        current += timedelta(days=1)

    return total

@app.route('/student_dashboard', methods=['GET','POST'])
def student_dashboard():
    data = None
    error = None

    if request.method == 'POST':
        roll = request.form.get('roll')
        student = users_collection.find_one({"id": roll})

        if not student:
            error = "Student Not Found"
        else:
            # Get the year of the student (e.g., "1", "2", "3", or "4")
            student_year = str(student.get('year'))
            
            # Find config specific to that student's year
            config = db.academic_config.find_one({"year": student_year})

            if not config:
                error = f"Academic calendar not configured for Year {student_year}"
            else:
                today_obj = date.today()
                start_obj = datetime.strptime(config['start_date'], "%Y-%m-%d").date()
                
                # Determine "Working days till today" 
                # (Clamp to end_date if today is after semester ends)
                end_obj = datetime.strptime(config['end_date'], "%Y-%m-%d").date()
                calculation_end = min(today_obj, end_obj)
                calculation_end_str = calculation_end.strftime("%Y-%m-%d")

                # 1. Working days till today
                working_till_today = get_working_days(
                    config['start_date'],
                    calculation_end_str,
                    config['holidays'],
                    config['weekly_off']
                )

                # 2. Total semester working days
                total_sem_working = get_working_days(
                    config['start_date'],
                    config['end_date'],
                    config['holidays'],
                    config['weekly_off']
                )

                # 3. Attended Days
                attended = attendance_collection.count_documents({"roll": roll})

                # 4. Percentage till today
                percent = round((attended / working_till_today) * 100, 2) if working_till_today else 0

                # 5. Leave Calculations (Target 80%)
                T = total_sem_working
                W = working_till_today
                A = attended
                required = int(np.ceil(0.8 * T))
                
                already_leave = W - A
                remaining_days = T - W
                need_more = required - A

                if need_more <= 0:
                    can_leave = remaining_days
                elif need_more > remaining_days:
                    can_leave = -1 # Impossible
                else:
                    can_leave = remaining_days - need_more

                data = {
                    "name": student['name'],
                    "dept": student.get('department','-'),
                    "year": student_year,
                    "attended": A,
                    "working_till_today": W,
                    "total_semester_days": T,
                    "percent": percent,
                    "already_leave": already_leave,
                    "can_leave": can_leave,
                    "required_attendance": required
                }

    return render_template("student_dashboard.html", data=data, error=error)

@app.route('/working_days', methods=['GET','POST'])
def working_days():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        # Capture the year this config belongs to
        target_year = request.form.get('target_year')
        
        db.academic_config.update_one(
            {"year": target_year},
            {"$set":{
                "year": target_year,
                "start_date": request.form['start'],
                "end_date": request.form['end'],
                "holidays": request.form.getlist('holidays'),
                "weekly_off": request.form.getlist('weekly')
            }},
            upsert=True
        )
        flash(f"Configuration for Year {target_year} saved successfully!", "success")

    # Fetch all configurations to show in the UI if needed
    configs = list(db.academic_config.find())
    return render_template("working_days.html", configs=configs)

def year_wise_chart(selected_date):
    chart_data = {}

    try:
        db_date = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%m_%d_%y")
    except:
        db_date = date.today().strftime("%m_%d_%y")

    for year in ['1', '2', '3', '4']:

        year_filter = {
            "$or": [
                {"year": year},
                {"year": str(year)},
                {"year": int(year)}
            ]
        }

        depts = users_collection.distinct("department", year_filter)

        data = []

        for d in depts:

            students = list(users_collection.find({
                "department": d,
                "$or": [
                    {"year": year},
                    {"year": str(year)},
                    {"year": int(year)}
                ]
            }))

            total = len(students)

            if total == 0:
                continue

            rolls = [s['id'] for s in students]

            present = attendance_collection.count_documents({
                "roll": {"$in": rolls},
                "date": db_date
            })

            percent = round((present / total) * 100, 1)

            data.append({
                "dept": d,
                "percent": percent
            })

        chart_data[year] = data

    return chart_data


def department_analytics_by_date(selected_date, selected_year=None):

    result = []

    try:
        db_date_format = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%m_%d_%y")
    except:
        db_date_format = date.today().strftime("%m_%d_%y")

    year_filter = {}
    if selected_year and selected_year != "":
        year_filter = {"year": selected_year}

    if year_filter:
        depts = users_collection.distinct("department", year_filter)
    else:
        depts = users_collection.distinct("department")

    for d in depts:

        student_query = {"department": d}

        if year_filter:
            student_query.update(year_filter)

        students = list(users_collection.find(student_query))

        total = len(students)
        if total == 0:
            continue

        rolls = [s['id'] for s in students]

        # 🔥 ONLY FORENOON SESSION COUNT
        present = attendance_collection.count_documents({
            "roll": {"$in": rolls},
            "date": db_date_format,
            "session": "forenoon"
        })

        result.append({
            "dept": d,
            "total": total,
            "present": present,
            "absent": total - present,
            "present_percent": round((present / total) * 100, 2)
        })

    return result

@app.route('/analytics', methods=['GET'])
def analytics():
    if session.get('role') != 'admin' and session.get('role') != 'faculty': # or if session.get('role') not in ['admin', 'faculty']: 
        flash("Unauthorized Access!")
        return redirect(url_for('home'))

    selected_date = request.args.get('date')
    selected_year = request.args.get('year')

    if not selected_date:
        selected_date = date.today().strftime("%Y-%m-%d")

    # ----- ROMAN YEAR FILTER -----
    student_filter = {}

    if selected_year and selected_year != "":
        student_filter = {"year": selected_year}
    # -----------------------------

    # 1. Department Summary
    table_data = department_analytics_by_date(selected_date, selected_year)

    # 2. Absent Students
    try:
        db_date_format = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%m_%d_%y")
    except:
        db_date_format = date.today().strftime("%m_%d_%y")

    all_eligible = list(users_collection.find(student_filter))

    present_rolls = attendance_collection.distinct(
        "roll",
        {"date": db_date_format}
    )

    absent_students = []

    for s in all_eligible:

        if s['id'] not in present_rolls:

            absent_students.append({
                "roll": s['id'],
                "name": s['name'],
                "dept": s.get('department', '-'),
                "year": s.get('year', '-')
            })

    return render_template(
        "analytics_dashboard.html",
        data=table_data,
        absent_students=absent_students,
        selected_date=selected_date,
        selected_year=selected_year
    )


@app.route('/grace_view')
def grace_view():

    date_filter = request.args.get('date')
    roll_filter = request.args.get('roll')

    query = {}

    # Date filter convert 2026-02-07 → 02_07_26
    if date_filter:
        try:
            db_date = datetime.strptime(date_filter, "%Y-%m-%d").strftime("%m_%d_%y")
            query["date"] = db_date
        except:
            pass

    if roll_filter:
        query["roll"] = roll_filter.strip()

    records = list(grace_attendance_collection.find(query).sort("date", -1))

    return render_template(
        "grace_attendance.html",
        records=records,
        selected_date=date_filter,
        selected_roll=roll_filter
    )

@app.route('/grace_request', methods=['GET', 'POST'])
def grace_request():
    # 1. Check Authentication
    user_email = session.get('email')
    if not user_email:
        return redirect(url_for('login'))

    # 2. Get Student Data
    student = users_collection.find_one({"email": user_email})
    if not student:
        return "Student Not Found", 404

    # 3. Handle Session Selection & Today's Date
    selected_session = request.form.get('session')
    today = date.today().strftime("%m_%d_%y")
    
    # 4. Fetch Grace Entry (Populates the Date/Time fields)
    grace = None
    if selected_session:
        grace = grace_attendance_collection.find_one({
            "roll": student['id'],
            "date": today,
            "session": selected_session
        })

    message = None

    # 5. Process Form Submission
    if request.method == 'POST' and request.form.get('reason'):
        # Safety Check: Ensure the grace data actually exists in DB before saving request
        # This prevents manual HTML manipulation of the 'readonly' fields
        if not grace or not grace.get('time') or not grace.get('date'):
            message = "Error: System Date and Time are mandatory. No valid log found for this session."
        
        else:
            # Prepare Data for Submission
            request_data = {
                "roll": student['id'],
                "name": student['name'],
                "email": student['email'],
                "phone": student.get('phone'),
                "department": student.get('department'),
                "year": student.get('year'),
                "mentor_name": student.get('mentor_name'),
                "mentor_email": student.get('mentor_email'),
                "session": selected_session,
                
                # These are pulled directly from the 'grace' DB record found above
                "date": grace['date'], 
                "time": grace['time'],
                
                "reason": request.form.get('reason'),
                "status": "pending",
                "applied_on": datetime.now()
            }

            # Insert into the Request Collection
            try:
                grace_request_collection.insert_one(request_data)
                message = "Request Submitted Successfully"
                # Optional: Reset variables so fields clear after success
                grace = None
                selected_session = None
            except Exception as e:
                message = f"Database Error: {str(e)}"

    return render_template(
        "grace_request.html",
        student=student,
        grace=grace,
        selected_session=selected_session,
        message=message
    )
 
@app.route('/grace_request_manage')
def grace_request_manage():

    requests = list(grace_request_collection.find().sort("applied_on", -1))

    return render_template(
        "grace_request_admin.html",
        requests=requests
    )

@app.route('/approve_grace/<id>')
def approve_grace(id):

    req = grace_request_collection.find_one({"_id": ObjectId(id)})

    if not req:
        return "Request Not Found"


    # ---- ADD TO MAIN ATTENDANCE ----
    attendance_collection.insert_one({

        "name": req['name'],
        "roll": req['roll'],

        "time": req['time'],
        "date": req['date'],

        "session": req['session']
    })

    # ---- UPDATE STATUS ----
    grace_request_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"status": "approved"}}
    )
    return redirect(url_for('grace_request_manage'))

@app.route('/zreject_grace/<id>')
def reject_grace(id):

    grace_request_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"status": "rejected"}}
    )

    return redirect(url_for('grace_request_manage'))



if __name__ == '__main__':
    app.run(debug=True)
