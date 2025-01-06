# Installation Guide

## Step 1: Install Python 3.10.10
### 1. Download Python 3.10.10
- Visit the official Python download page at [Python 3.10.10](https://www.python.org/downloads/release/python-31010/) to download the installer for Windows.
- Click on "Windows installer (64-bit)" to download the installer suitable for most users.

### 2. Run the Installer
- Once downloaded, run the installer.
- Ensure to check the box that says "Add Python 3.10 to PATH" at the bottom of the installer window to set Python in your system PATH.
- Click on "Install Now".

## Step 2: Set Up Python Environment
### 1. Open Command Prompt in Code Folder
- Navigate to the folder containing your Python code.
- Hold down `Shift`, right-click in the folder, and select "Open in Terminal" or "Open PowerShell window here".

### 2. Install Required Python Packages
- In the command prompt or PowerShell window, type the following command and press `Enter`:
```
pip install -r requirements.txt
```
- This will install all the packages listed in your `requirements.txt` file.

## Step 3: Install Tesseract OCR
### 1. Download Tesseract
- Go to the official Tesseract at [UB Mannheim's Tesseract page](https://github.com/UB-Mannheim/tesseract/wiki) as it provides precompiled binaries for Windows.
- Download the installer for the latest version compatible with your system (e.g., `tesseract-ocr-w64-setup-v5.0.0.20210503.exe` for 64-bit systems).

### 2. Run the Tesseract Installer
- Launch the downloaded installer.
- Complete the installation.

### 3. Add Tesseract OCR Installation Directory to Your System PATH
#### - Find the Installation Directory
- Tesseract is typically installed in a directory like `C:\Program Files\Tesseract-OCR` or a similar path.

#### - Add the Directory to System PATH
- Open the Start Search by clicking the Start button or pressing `Win`.
- Type `env` or `environment variables` and select "Edit the system environment variables" or "Edit environment variables for your account".
- Click on the "Environment Variables..." button.
- Find the `Path` variable under "System variables" or "User variables". Select it and click "Edit...".
- Click "New" and enter the path to the Tesseract installation directory.
- Click "OK" to close all dialogs.

#### - Verify the Change
- Open a new command prompt.
- Type `tesseract --version` and press `Enter`. You should see the version information if Tesseract is correctly added to your PATH.

## Step 4: Run the Script
### 1. Launch the F124 Game
- Start the F124 game application.

### 2. Execute the Script
- In the command prompt or PowerShell window, type the following command and press Enter:
```
python main_mss.py
```