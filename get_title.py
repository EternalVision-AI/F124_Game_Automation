import win32gui
import win32con
import win32process
import os

def enum_windows_callback(hwnd, windows):
    """
    Callback function for win32gui.EnumWindows.
    Appends window handles and titles to the windows list.
    """
    if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
        windows.append((hwnd, win32gui.GetWindowText(hwnd)))

def get_active_windows():
    """
    Retrieves a list of tuples containing window handles and their titles.
    
    Returns:
        List of tuples: [(hwnd1, title1), (hwnd2, title2), ...]
    """
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    return windows

def save_windows_to_file(windows, filename="active_windows.txt"):
    """
    Saves the list of windows to a text file.
    
    Args:
        windows (list): List of tuples containing window handles and titles.
        filename (str): Name of the file to save the window titles.
    """
    with open(filename, "w", encoding="utf-8") as file:
        for hwnd, title in windows:
            file.write(f"{hwnd}: {title}\n")
    print(f"Window titles saved to {filename}")

def print_windows(windows):
    """
    Prints the list of windows to the console.
    
    Args:
        windows (list): List of tuples containing window handles and titles.
    """
    print(f"{'Handle':<15} Title")
    print("-" * 50)
    for hwnd, title in windows:
        print(f"{hwnd:<15} {title}")

def main():
    # Retrieve all active window titles
    windows = get_active_windows()
    
    # Print to console
    print_windows(windows)
    
    # Optionally, save to a file
    save_option = input("Would you like to save the window titles to a file? (y/n): ").strip().lower()
    if save_option == 'y':
        filename = input("Enter the filename (default is active_windows.txt): ").strip()
        if not filename:
            filename = "active_windows.txt"
        save_windows_to_file(windows, filename)

if __name__ == "__main__":
    main()
