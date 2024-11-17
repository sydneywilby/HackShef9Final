import streamlit as st
from pymongo import MongoClient
from datetime import datetime, timedelta
import pytz
import time

# MongoDB connection settings
MONGODB_URI = "mongodb+srv://sydneywilby:csTLuCfjkE3UMd1y@firebotdetection.oiwyq.mongodb.net/?retryWrites=true&w=majority&appName=FireBotDetection"
DATABASE_NAME = "fire_detection"
COLLECTION_NAME = "alerts"

# Connect to MongoDB
@st.cache_resource
def get_mongo_client():
    client = MongoClient(MONGODB_URI)
    return client

# Get the count of recent alerts (within the last 5 minutes)
def get_recent_alerts_count():
    client = get_mongo_client()
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    # Calculate the time 5 minutes ago from now
    five_minutes_ago = datetime.now(pytz.UTC) - timedelta(seconds=30)
    
    # Query MongoDB to count alerts with timestamps greater than 5 minutes ago
    alert_count = collection.count_documents({"timestamp": {"$gt": five_minutes_ago}})
    return alert_count

# Function to get the background color based on alert count
def get_alert_color(alert_count):
    # The color gets more red as the count increases
    if alert_count == 0:
        return "#00FF00"  # Green for zero alerts
    elif alert_count < 5:
        return "#FFFF00"  # Yellow for a low number of alerts
    elif alert_count < 10:
        return "#FFA500"  # Orange for moderate alerts
    else:
        return "#FF0000"  # Red for a high number of alerts

# Streamlit UI
st.set_page_config(page_title="Fire Detection Dashboard", page_icon="🔥", layout="centered")

st.title("🚨 Fire Detection System Dashboard")
st.header("Real-Time Alert Monitoring")

# Get the recent alert count and corresponding color
recent_alerts_count = get_recent_alerts_count()
bg_color = get_alert_color(recent_alerts_count)

# Display the metric with dynamic background color
st.markdown(
    f"""
    <div style="padding: 20px; background-color: {bg_color}; border-radius: 10px; text-align: center;">
        <h2 style="color: white;">Active Alerts</h2>
        <h1 style="font-size: 60px; color: white;">{recent_alerts_count}</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Add an auto-refresh every second
time.sleep(1)
st.rerun()
