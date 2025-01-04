import cv2
import numpy as np
from ctypes import windll
import win32gui
import win32ui
import time
from threading import Thread
from contextlib import contextmanager
from paddleocr import PaddleOCR
import pytesseract

from custom_utils import identify_screen, api_screen

@contextmanager
def gdi_resource_management(hwnd):
    # Acquire resources
    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    
    try:
        yield hwnd_dc, mfc_dc, save_dc, bitmap
    finally:
        # Ensure resources are released
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
   
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
            # raise RuntimeError(f"Unable to acquire screenshot! Result: {result}")
        
        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)

    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
    img = np.ascontiguousarray(img)[..., :-1]  # make image C_CONTIGUOUS and drop alpha channel
    # cv2.imwrite("test.png", img)
    return img

def capture_and_process_image(window_name, img_count, ocr):
    img = capture_win_alt(window_name)
    if img is not None:
        extracted_text = pytesseract.image_to_string(preprocess_image(img), lang='eng')
        # print(extracted_text)
        extracted_text = extracted_text.upper()
        screen_title = identify_screen(extracted_text)
        print("----------------------------------------------------------")
        print(screen_title)
        api_screen(screen_title)
        if img is not None:
            # Create the folder, ensuring no error is raised if it already exists
            os.makedirs("screen_images", exist_ok=True)
            cv2.imwrite(f"screen_images/{img_count}_{screen_title}.png", img)
    return img_count + 1

import psutil
import time

# Start monitoring resources
start_time = time.time()
process = psutil.Process()

def monitor_game(window_name, ocr):
    img_count = 1
    while True:
        img_count = capture_and_process_image(window_name, img_count, ocr)
        
        cpu_percent = process.cpu_percent(interval=1)
        memory_info = process.memory_info()
        print(f"CPU Usage: {cpu_percent}%")
        print(f"Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")
        print("----------------------------------------------------------")
        time.sleep(2)

def main():
    window_name = 'F1® 24'
    # Start monitoring in a separate thread
    game_thread = Thread(target=monitor_game, args=(window_name, None))
    game_thread.start()
    game_thread.join()

if __name__ == "__main__":
    main()