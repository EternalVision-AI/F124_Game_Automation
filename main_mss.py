import cv2
import numpy as np
from ctypes import windll
import win32gui
import win32ui
import time
from threading import Thread
from contextlib import contextmanager
from paddleocr import PaddleOCR

from custom_utils import ocr_screen

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
        _, screen_title = ocr_screen(img, ocr)
        print("----------------------------------------------------------")
        print(screen_title)
        # if processed_img is not None:
        #     cv2.imwrite(f"screen_images/{img_count}.png", processed_img)
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
    ocr = PaddleOCR(
        lang="en",
        use_gpu=False,
        cpu_threads=6,  # Reduced threads to avoid excessive resource consumption
        enable_mkldnn=True,
        det_db_score_mode="slow",  # Use slow mode for better accuracy
        use_angle_cls=False,  # Enable angle classification
        det_limit_side_len=5880,  # Increase detection size limit
        det_db_box_thresh=0.1,  # Lower threshold to detect more text
        det_db_thresh=0.1,  # Lower threshold for text detection
        rec_batch_num=4,  # Increase batch size for recognition
        det_db_unclip_ratio=2,  # Increase unclip ratio to expand text regions
        max_text_length=100,
        drop_score=0.1,
    )
    window_name = 'F1® 24'
    window_name = '(13).png'
    # Start monitoring in a separate thread
    game_thread = Thread(target=monitor_game, args=(window_name, ocr))
    game_thread.start()
    game_thread.join()

if __name__ == "__main__":
    main()