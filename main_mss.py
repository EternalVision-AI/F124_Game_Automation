import json
import os
import time
from contextlib import contextmanager
from ctypes import windll
from threading import Thread
import cv2
import numpy as np
import psutil
import pytesseract
import win32gui
import win32ui
from flask import Flask, jsonify, request
from custom_utils import api_screen, identify_screen

# Global Variable
isToggle = True

# Start monitoring resources
start_time = time.time()
process = psutil.Process()

# Flask App
app = Flask(__name__)

@app.route("/api/togglevision", methods=["POST"])
def toggle_vision():
    global isToggle
    # Get the JSON data from the request
    data = request.get_json()
    # Check if the 'ToggleVision' key exists in the request data
    if "ToggleVision" not in data:
        return jsonify({"error": "Missing 'ToggleVision' in request data"}), 400
    # Get the value of 'ToggleVision'
    toggle_vision_value = data["ToggleVision"]
    # Perform your logic based on the toggle value (e.g., printing or toggling some system functionality)
    print(f"Received ToggleVision: {toggle_vision_value}")
    isToggle = toggle_vision_value
    # Respond back with a success message

    return jsonify({"message": f"ToggleVision set to {toggle_vision_value} successfully!"}), 200


@contextmanager
def gdi_resource_management(hwnd):
    # Acquire resources
    hwnd_dc = None
    mfc_dc = None
    save_dc = None
    bitmap = None
    try:
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        if not hwnd_dc:
            raise RuntimeError("Failed to get window DC")

        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        
        bitmap = win32ui.CreateBitmap()
        # Retrieve window dimensions
        rect = win32gui.GetClientRect(hwnd)
        width, height = rect[2] - rect[0], rect[3] - rect[1]

        # Create compatible bitmap
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        yield hwnd_dc, mfc_dc, save_dc, bitmap
    finally:
        # Ensure resources are released safely
        if bitmap:
            win32gui.DeleteObject(bitmap.GetHandle())
        if save_dc:
            save_dc.DeleteDC()
        if mfc_dc:
            mfc_dc.DeleteDC()
        if hwnd_dc:
            win32gui.ReleaseDC(hwnd, hwnd_dc)


# Function to load configuration from the JSON file
def load_config(config_path="config.json"):
    with open(config_path, "r") as config_file:
        config_data = json.load(config_file)
    return config_data

# Convert the image to gray scale
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def capture_win_alt(window_name: str):
    windll.user32.SetProcessDPIAware()
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        print(f"Window '{window_name}' not found. Skipping capture.")
        return None  # Exit the function if window is not found
    if win32gui.IsIconic(hwnd):
        print(f"Window '{window_name}' is minimized. Skipping capture.")
        return None
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bottom - top

    with gdi_resource_management(hwnd) as (hwnd_dc, mfc_dc, save_dc, bitmap):
        bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bitmap)

        result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)

        if not result:
            print(f"Window is minimized")
            return None
        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape(
        (bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4)
    )
    img = np.ascontiguousarray(img)[..., :-1]  # make image C_CONTIGUOUS and drop alpha channel
    return img

def capture_and_process_image(window_name, img_count):
    img = capture_win_alt(window_name)
    if img is not None:
        extracted_text = pytesseract.image_to_string(preprocess_image(img), lang="eng")
        extracted_text = extracted_text.upper()
        print(extracted_text)
        screen_title = identify_screen(extracted_text)
        print("----------------------------------------------------------")
        print(screen_title)
        api_screen(screen_title)
        if img is not None:
            # Create the folder, ensuring no error is raised if it already exists
            os.makedirs("screen_images", exist_ok=True)
            cv2.imwrite(f"screen_images/{img_count%20}_{screen_title}.png", img)
    return img_count + 1


def monitor_game(window_name):
    img_count = 1
    while True:
        print("----------------------------------------------------------")
        if isToggle:
            print("Caturing...")
            img_count = capture_and_process_image(window_name, img_count)
        else:
            print("No capturing...")
        process.cpu_percent(interval=1)
        print("----------------------------------------------------------")
        process.memory_info()
        time.sleep(2)

def run_flask(port):
    print("Running Flask App...")
    # Run the Flask app with use_reloader=False to avoid issues when running in threads
    app.run(port=port, use_reloader=False)  # Prevent Flask from using reloader in threads

def main():
    # Load configuration from JSON file
    config = load_config("config.json")  # You can replace the filename if needed
    
    # Run flask app in a separate thread
    port = config["port_receive"]
    flask_thread = Thread(target=run_flask, args=(port,))
    flask_thread.start()

    # Start monitoring in a separate thread
    window_name = "F1® 24"
    game_thread = Thread(target=monitor_game, args=(window_name,))
    game_thread.start()
    game_thread.join()


if __name__ == "__main__":
    main()