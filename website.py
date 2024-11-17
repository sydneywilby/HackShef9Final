import json
from os import environ as env
from urllib.parse import quote_plus, urlencode
from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from flask import Flask, redirect, render_template, session, url_for, jsonify,request,send_from_directory, abort
import os
import base64
from pymongo import MongoClient
from datetime import datetime, timedelta
import cv2
import numpy as np

from keras.models import load_model

print("Starting up...")

# Load the trained model
model = load_model('fire_detection_model.keras')

# Load environment variables from .env file
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = env.get("SECRET_KEY")

# MongoDB setup
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
uri = env.get("MONGODB_URI")
if not uri:
    raise ValueError("No MONGODB_URI environment variable set")
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


db = client["fire_detection"]
cameras_collection = db["cameras"]
alerts_collection = db["alerts"]

# Initialize OAuth client for Auth0
oauth = OAuth(app)

# Register Auth0 with OAuth
oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid profile email",  # Request openid, profile, and email
    },
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration',  # This provides jwks_uri
)


# Controllers / Routes
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/callback", methods=["GET", "POST"])
def callback():
    # Auth0 callback to authorize access token
    token = oauth.auth0.authorize_access_token()
    session["user"] = token
    return redirect("/dashboard")  # Redirect to home page after successful login


@app.route("/login")
def login():
    if "user" in session:
        return redirect(url_for("dashboard"))  # If user is already logged in, redirect to home
    # Redirect user to Auth0 login page
    return oauth.auth0.authorize_redirect(
        redirect_uri=url_for("callback", _external=True)  # Redirect to callback route after login
    )


@app.route("/logout")
def logout():
    # Step 1: Clear the local session
    session.clear()

    # Step 2: Redirect to the Auth0 logout URL
    auth0_domain = os.getenv("AUTH0_DOMAIN")
    client_id = os.getenv("AUTH0_CLIENT_ID")
    return_to = url_for("home", _external=True)  # Redirect back to your homepage after logout

    # Construct the Auth0 logout URL
    auth0_logout_url = (
        f"https://{auth0_domain}/v2/logout?"
        + urlencode(
            {
                "returnTo": return_to,
                "client_id": client_id,
            },
            quote_via=quote_plus,
        )
    )

    return redirect(auth0_logout_url)


# Secured route example
@app.route("/dashboard")
def profile():
    # Check if user is logged in (i.e., session contains user info)
    if "user" not in session:
        return redirect(url_for("login"))  # If not logged in, redirect to login

    cameras = list(cameras_collection.find())
    alerts = list(alerts_collection.find())
    return render_template("dashboard.html", user=session["user"],cameras = cameras,alerts = alerts)  # Display profile page


@app.route("/camera/live/<int:camera_id>", methods=['GET'])
def view_camera(camera_id):
    if "user" not in session:
        return redirect(url_for("login"))
    
    if camera_id is None:
        return "Camera ID is required", 400
    
    return render_template("camera.html", user=session["user"],camera_id = camera_id)  # Display profile page

@app.route('/camera/getview/<camera_id>', methods=['GET'])
def get_camera_image(camera_id):
    if "user" not in session:
        return redirect(url_for("login"))
    
    camera = cameras_collection.find_one({"camera_id": int(camera_id)})
    
    if camera:
        return camera["image"], 200
    else:
        abort(404, description="Image not found")

@app.route('/camera/view/<camera_id>', methods=['GET'])
def view_camera_image(camera_id):
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("camerastream.html", user=session["user"],camera_id = camera_id)

@app.route('/upload/camera/<int:camera_id>', methods=['POST'])
def upload(camera_id):
    if "user" not in session:
        return redirect(url_for("login"))

    if camera_id is None:
        return "Camera ID is required", 400

    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Process base64 image
        image_data = data['image']
        image_bytes = base64.b64decode(image_data.split(',')[1])
        latitude = float(data['location']['latitude'])
        longitude = float(data['location']['longitude'])

        # Save image and update database
        cameras_collection.update_one(
            {"camera_id": camera_id},
            {
                "$set": {
                    "image": image_bytes,
                    "last_updated": datetime.now(),
                    "latitude": latitude,
                    "longitude": longitude
                }
            },
            upsert=True
        )

        return jsonify({
            'message': 'Image processed successfully',
        }), 200

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/predict/<camera_id>', methods=['POST'])
def predict_fire(camera_id):
    camera = cameras_collection.find_one({"camera_id": int(camera_id)})
    if camera and camera["image"]:
        # Convert bytes to numpy array for prediction
        nparr = np.frombuffer(camera["image"], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run prediction
        img_resized = cv2.resize(img, (224, 224)) / (255.0 * 1.5)
        pred = model.predict(np.expand_dims(img_resized, axis=0))
        pred_prob = float(pred[0][0])

        google_maps_url = f"https://maps.google.com/maps?width=100%25&height=600&hl=en&q={camera['latitude']},{camera['longitude']}&t=&z=14&ie=UTF8&iwloc=B&output=embed"

        if pred_prob > 0.9:
            # Fire detected
            # Save the alert to the MongoDB database
            now = datetime.now()
            alerts_collection.insert_one({
                "camera_id": camera_id,
                "timestamp": now,
                "fire_prob": pred_prob,
                "latitude": camera["latitude"],
                "longitude": camera["longitude"]
            })

        return jsonify({"fire_prob": pred_prob, "google_maps_url": google_maps_url})

    return jsonify({"error": "Image not found"}), 404

@app.route('/camera/delete/<camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    if "user" not in session:
        return redirect(url_for("login"))  # If not logged in, redirect to login
    
    # Check if the camera_id exists in the MongoDB database
    camera = cameras_collection.find_one({"camera_id": int(camera_id)})
    
    if camera:
        # Camera found, delete it
        cameras_collection.delete_many({"camera_id": int(camera_id)})
        return {"message": f"Camera with ID {camera_id} has been deleted."}, 200
    else:
        # Return a 404 error if the camera does not exist in the database
        abort(404, description="Camera not found in database")

@app.route('/alerts')
def alerts():
    if "user" not in session:
        return redirect(url_for("login"))  # If not logged in, redirect to login
    # Simulate fire detection alert

    now = datetime.now()
    # Get all alerts from the MongoDB database that are less than than 5 minutes old
    five_minutes_ago = now - timedelta(minutes=5)
    alerts = list(alerts_collection.find({"timestamp": {"$gt": five_minutes_ago}}))[::-1]

    # Convert ObjectId to string
    for alert in alerts:
        alert["_id"] = str(alert["_id"])

    return jsonify(alerts)


if __name__ == "__main__":
    # Run Flask app on the specified port
    app.run(host="0.0.0.0")