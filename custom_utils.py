import json
import os
import cv2
import numpy as np
import requests

# Function to load configuration from the JSON file
def load_config(config_path='config.json'):
    with open(config_path, 'r') as config_file:
        config_data = json.load(config_file)
    return config_data

# Load configuration from JSON file
config = load_config('config.json')  # You can replace the filename if needed

# Get values from the config file
port = config['port_send']

def api_screen(screen_id):
    # URL of the API endpoint
    url = f"http://127.0.0.1:{port}/api/screenstatus"
    
    # Payload data
    payload = {
        "screen_id": screen_id
    }
    
    # Set headers (optional; include if required by your API)
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Send the POST request
        response = requests.post(url, json=payload, headers=headers, verify=False)
        
        # Check if the request was successful
        if response.status_code == 200:
            print(f"Request: {screen_id} successful!")
            print(f"{response.text}")
            # return r  # Returns the JSON response
        else:
            print(f"Request: {screen_id} failed with status code:", response.status_code)
            print("Error:", response.text)
            # return None  # Return None in case of failure
    except requests.RequestException as e:
        print("An error occurred:", e)
        # return None

def identify_screen(text):
    screens = {
        "SCREEN_START": ("MMM"),
        "SCREEN_MAIN_MENU": ("LEAGUE", "THEATRE"),
        "SCREEN_SETTING_MENU": ("HELP", "SETTINGS"),
        "SCREEN_GRAPHIC_SETTING": ("GRAPHICS SETTINGS", "ADVANCED SETUP"),
        "SCREEN_WORLD_MENU": ("TROPHIES", "CUSTOMISATION"),
        "SCREEN_RACE_MENU": ("TIME", "CONNECTION"),
        "SCREEN_TEAM_SELECT": ("TEAM SELECT", "ADVANCE"),
        "SCREEN_TRACK_SELECT": ("TIME TRIAL", "SELECT EVENT"),
        "SCREEN_TRACK_SELECT_GRANDPRIX": ("CUSTOM CHAMPIONSHIP", "ROUND"),
        "SCREEN_NO_NETWORK": ("NO NETWORK", "SIGNED IN"),
        "SCREEN_CONFIRM_CHANGE": ("CONFIRM", "CHANGES"),
    }
    for screen, words in screens.items():
        if all(word in text for word in words):
            return screen
    return "SCREEN_OTHER"
