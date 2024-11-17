import json
from os import environ as env
import sys
import time
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

# get the camera id from the command line as an argument
camera_id = int(sys.argv[1])

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

is_evacuating = False
evacuation_start_time = datetime.now()

def activate_suppression_system():
    print("Activating suppression system for camera " + str(camera_id))
    time.sleep(20)
    print("Suppression system stopped")
    print("Normal operation resumed")
    is_evacuating = False

def get_alerts_in_range(start):
    return list(alerts_collection.find({"timestamp": {"$gte": start}},{"camera_id": camera_id}))

def start_evacuation():
    if is_evacuating:
        return
    print("Starting evacuation for camera " + str(camera_id))
    is_evacuating = True
    evacuation_start_time = datetime.now()

    time.sleep(60)
    if is_evacuating and len(get_alerts_in_range(datetime.now() - timedelta(minutes=1))) > 20:
        activate_suppression_system()

def stop_evacuation():
    if not is_evacuating:
        return
    print("Stopping evacuation for camera " + str(camera_id))

def get_camera_status():
    camera = cameras_collection.find_one({"camera_id": camera_id})
    if not camera:
        sys.exit("Camera not found")
    if camera["last_updated"] > datetime.now() - timedelta(minutes=1):
        return "online"
    else:
        return "offline"

def checkup():
    if get_camera_status() == "offline":
        print("Camera " + str(camera_id) + " is offline")
    if len(get_alerts_in_range(datetime.now() - timedelta(minutes=1))) > 20:
        start_evacuation()
    else:
        stop_evacuation()

# run the checkup function every 10 seconds
while True:
    checkup()
    print("Checked up on camera " + str(camera_id))
    time.sleep(30)

