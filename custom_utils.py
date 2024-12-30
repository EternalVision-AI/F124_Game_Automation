import os
from paddleocr import PaddleOCR
import cv2
import numpy as np
import requests

def api_screen(screen_id, marble_id, lap, time):
    # URL of the API endpoint
    url = "https://p1su5ofsta.execute-api.me-south-1.amazonaws.com/dev?action=raceLive"
    
    # Payload data
    payload = {
        "screen_id": screen_id,
        "marble_id": marble_id,
        "lap": lap,
        "time": time
    }
    
    # Set headers (optional; include if required by your API)
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Send the POST request
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            print(f"Request: {screen_id, marble_id, lap, time} successful!")
            print(f"{response.text}")
            # return r  # Returns the JSON response
        else:
            print(f"Request: {screen_id, marble_id, lap, time} failed with status code:", response.status_code)
            print("Error:", response.text)
            # return None  # Return None in case of failure
    except requests.RequestException as e:
        print("An error occurred:", e)
        # return None

def identify_screen(text):
    screens = {
        "SCREEN_START": ("PRESSANY", "BUTTON"),
        "SCREEN_MAIN_MENU": ("LEAGUE", "THEATRE"),
        "SCREEN_SETTING_MENU": ("HELP", "SETTINGS"),
        "SCREEN_GRAPHIC_SETTING": ("GRAPHICS SETTINGS", "ADVANCED SETUP"),
        "SCREEN_WORLD_MENU": ("TROPHIES", "CUSTOMISATION"),
        "SCREEN_RACE_MENU": ("TIME", "CONNECTION"),
        "SCREEN_TEAM_SELECT": ("TEAM SELECT", "ADVANCE"),
        "SCREEN_TRACK_SELECT": ("TIME TRIAL", "SELECT EVENT"),
    }
    for screen, words in screens.items():
        if all(word in text for word in words):
            return screen
    return "SCREEN_OTHER"

def ocr_screen(img, ocr):
    if img is None:
        print(f"Failed to read image: {img}")
        return None

    extracted_text = []
    # Perform OCR
    result = ocr.ocr(img, cls=True)

    # Draw rectangles and text if result is not None
    if result is not None:
        for line in result:
            if not line:
                continue
            for word_info in line:
                # word_info is typically [ [ [x0, y0], [x1, y1], [x2, y2], [x3, y3] ], ("text", confidence_score) ]
                if len(word_info) < 2:
                    continue  # Skip incomplete word info

                box, (text, score) = word_info
                if len(box) < 4:
                    continue  # Skip incomplete bounding box info
                # Append text to the list
                extracted_text.append(text)
                # Coordinates of top-left and bottom-right corners
                x0, y0 = box[0]
                x1, y1 = box[2]
                x0 = int(x0)
                y0 = int(y0)
                x1 = int(x1)
                y1 = int(y1)
    
                # Generate a random color for each rectangle
                # color = tuple(np.random.randint(0, 255, 3).tolist())
                
                # Draw rectangle around text
                # cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
                
                # Put text above the rectangle
                # cv2.putText(img, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    result_str = " ".join(extracted_text)
    print(result_str)
    screen_title = identify_screen(result_str)
    # Attempt to make the image writable
    # if not img.flags.writeable:
    #     writable_img = img.copy()
    # else:
    #     writable_img = img
    # cv2.putText(writable_img, screen_title, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # cv2.imshow("Game Screen", cv2.resize(writable_img, (800, 600)))
    # cv2.waitKey(0)
    # return True
    # return writable_img, screen_title
    return None, screen_title