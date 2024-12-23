import win32gui
import win32ui
import win32con
from ctypes import windll
from PIL import Image
from custom_utils import ocr_screen, api_screen
import numpy as np
import cv2
import time
import psutil
import logging
from paddleocr import PaddleOCR

def set_low_priority():
    """
    Sets the process priority to IDLE to minimize CPU impact.
    """
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.IDLE_PRIORITY_CLASS)
        logging.info("Process priority set to IDLE.")
    except Exception as e:
        logging.error(f"Failed to set process priority: {e}")

def activate_window(hwnd):
    """
    Activates (restores and brings to foreground) the specified window.
    
    Args:
        hwnd (int): Handle to the window.
    
    Returns:
        bool: True if the window was successfully activated, False otherwise.
    """
    try:
        # Restore the window if it is minimized
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        print(f"Window handle {hwnd} restored.")
        
        # Bring the window to the foreground
        win32gui.SetForegroundWindow(hwnd)
        print(f"Window handle {hwnd} brought to foreground.")
        
        # Optional: Wait briefly to ensure the window has time to restore
        time.sleep(1)
        
        return True
    except Exception as e:
        print(f"Failed to activate window handle {hwnd}: {e}")
        return False

def capture_window(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    # Check if the window handle is valid
    if hwnd == 0:
        print(f"Window '{window_name}' not found. Skipping capture.")
        return None  # Exit the function if window is not found
    if win32gui.IsIconic(hwnd):
        print(f"Window '{window_name}' is minimized. Skipping capture.")
        activated = activate_window(hwnd)
        if not activated:
            print(f"Could not activate window '{window_name}'. Skipping capture.")
            return None  # Exit if activation failed
        return None
    # Uncomment the following line if you use a high DPI display or >100% scaling size
    # windll.user32.SetProcessDPIAware()

    try:
        # Change the line below depending on whether you want the whole window
        # or just the client area. 
        #left, top, right, bot = win32gui.GetClientRect(hwnd)
        left, top, right, bot = win32gui.GetWindowRect(hwnd)
        w = right - left
        h = bot - top

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

        saveDC.SelectObject(saveBitMap)

        # Change the line below depending on whether you want the whole window
        # or just the client area. 
        #result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        img = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        if result == 1:
            #PrintWindow Succeeded
            # img.save("test.png")
             # Ensure image is in RGB format
            img = img.convert('RGB')
            # Convert to NumPy array
            open_cv_image = np.array(img)
            # Convert RGB to BGR
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            return open_cv_image
        else:
            return None
    except Exception as e:
        print(f"An error occurred while capturing the window '{window_name}': {e}")
        return False




if __name__ == "__main__":
    window_name = 'LDPlayer'
    set_low_priority()
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)  # Enable GPU if available
    while True:    
        img = capture_window(window_name)
        if img is not None:
            ocr_screen(img, ocr)
        time.sleep(3)