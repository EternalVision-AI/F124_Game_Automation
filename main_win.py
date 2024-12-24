import win32gui
import win32ui
import win32con
import win32api
import mss
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

def set_windowed_mode(window_title):
    # Find the window handle based on the window title
    hwnd = win32gui.FindWindow(None, window_title)
    
    if hwnd == 0:
        print(f"Error: Window '{window_title}' not found.")
        return False
    
    print(f"Window '{window_title}' found with handle: {hwnd}")
    
    # Check if the window is minimized
    if win32gui.IsIconic(hwnd):
        print(f"Window '{window_title}' is minimized. Restoring window.")
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        time.sleep(1)  # Wait for the window to restore
    
    # Get the current window rectangle
    rect = win32gui.GetWindowRect(hwnd)
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    print(f"Original window size: {width}x{height}")
    
    # Get the current window style
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
    print(f"Original window style: {style}")
    
    # Modify the style to remove fullscreen attributes
    style &= ~win32con.WS_POPUP
    style |= win32con.WS_OVERLAPPEDWINDOW
    print(f"Modified window style: {style}")
    
    win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
    
    # Optional: Make the window topmost temporarily
    win32gui.SetWindowPos(
        hwnd,
        win32con.HWND_TOPMOST,
        0,
        0,
        0,
        0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
    )
    
    # Get the monitor info to center the window
    monitor = win32api.MonitorFromWindow(hwnd)
    monitor_info = win32api.GetMonitorInfo(monitor)
    monitor_area = monitor_info['Monitor']
    monitor_width = monitor_area[2] - monitor_area[0]
    monitor_height = monitor_area[3] - monitor_area[1]
    
    # Calculate position to center the window
    x = monitor_area[0] + (monitor_width - width) // 2
    y = monitor_area[1] + (monitor_height - height) // 2
    
    print(f"Setting window position to ({x}, {y}) with size ({width}x{height})")
    
    # Resize and reposition the window
    win32gui.SetWindowPos(
        hwnd,
        None,
        x,
        y,
        width,
        height,
        win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED
    )
    
    # Remove the topmost flag
    win32gui.SetWindowPos(
        hwnd,
        win32con.HWND_NOTOPMOST,
        x,
        y,
        width,
        height,
        win32con.SWP_NOZORDER | win32con.SWP_FRAMECHANGED
    )
    
    print(f"Window '{window_title}' set to windowed mode.")
    return True

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
        # win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
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
        # activated = set_windowed_mode(window_name)
        # if not activated:
        #     print(f"Could not activate window '{window_name}'. Skipping capture.")
        #     return None  # Exit if activation failed
        # return None
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
            img.save("test.png")
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
    window_name = 'Everything'
    # window_name = 'Counter-Strike'
    window_name = 'F1® 24'
    set_low_priority()
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)  # Enable GPU if available
    while True:    
        img = capture_window(window_name)
        if img is not None:
            ocr_screen(img, ocr)
        time.sleep(3)