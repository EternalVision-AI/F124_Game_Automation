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
- Follow the on-screen instructions. When prompted, make sure to select the option to "Add Tesseract to the system PATH".
- Complete the installation.

## Completion
- You have now successfully installed Python, set up your Python environment with required packages, and installed Tesseract OCR with the necessary system configuration. You are ready to start using your setup for development.

## Step 4: Run the Script
### 1. Launch the F124 Game
- Start the F124 game application.

### 2. Execute the Script
- In the command prompt or PowerShell window, type the following command and press Enter:
```
python main_mss.py
```