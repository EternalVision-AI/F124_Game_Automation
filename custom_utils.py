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
  if "PRESSANY" in text and "BUTTON" in text:
    return "SCREEN_START"
  elif "CAREER" in text and "SETTINGS" in text:
    return "SCREEN_MAIN_MENU"
  elif "F10HELP" in text and "F2SETTINGSFILTER" in text:
    return "SCREEN_SETTING_MENU"
  elif "GRAPHICS SETTINGS" in text and "F2SETTINGSFILTER" in text:
    return "SCREEN_GRAPHIC_SETTING"
  elif "PLAY" in text and "CUSTOMISATION" in text:
    return "SCREEN_WORLD_MENU"
  elif "OFFLINE" in text and "GRAND PRIXTM" in text:
    return "SCREEN_RACE_MENU"
  elif "TEAM SELECT" in text and "ADVANCE" in text:
    return "SCREEN_TEAM_SELECT"
  elif "TIME TRIAL" in text and "SELECT EVENT" in text:
    return "SCREEN_TRACK_SELECT"
  else:
    return "SCREEN_OTHER"

def ocr_screen(img, ocr):
    if img is None:
        print(f"Failed to read image: {image_path}")
        return False

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
                color = tuple(np.random.randint(0, 255, 3).tolist())
                
                # Draw rectangle around text
                # cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
                
                # Put text above the rectangle
                # cv2.putText(img, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    screen_title = identify_screen(extracted_text)
    cv2.putText(img, screen_title, (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
    
    cv2.imshow("Game Screen", img)
    cv2.waitKey(1)
    # return True


def process_images_in_folder(input_folder):
    # Ensure the input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return

    # Process each image file in the folder
    for image_file in os.listdir(input_folder):
        # Check if the file is an image (basic check by extension)
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)
        success = ocr_screen(img)
        if not success:
            print(f"Skipping file: {image_file} due to read errors.")


# Example usage:
# process_images_in_folder('./images')
