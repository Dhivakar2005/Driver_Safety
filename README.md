# 🚗 Driver Safety System

A smart **Driver Safety Monitoring System** designed to enhance road safety using real-time video analysis and machine learning.  
This project detects **driver drowsiness**, **distraction**, and other unsafe behaviors to prevent accidents and promote responsible driving.

---

## 📋 Table of Contents
- [About the Project](#-about-the-project)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model & Dataset](#-model--dataset)
- [Results](#-results)
- [Future Improvements](#-future-improvements)
- [Contributors](#-contributors)
- [License](#-license)

---

## 🔍 About the Project
The **Driver Safety System** monitors the driver’s face and eyes in real time to detect signs of fatigue, distraction, or unsafe behavior.  
Using computer vision and deep learning, it analyzes live video feeds from a camera mounted inside a vehicle and triggers an alert when unsafe conditions are detected.

### 🎯 Objective
To reduce accidents caused by **drowsy or distracted driving** through an automated, real-time alerting system.

---

## 🌟 Features
- 🧠 **Real-time Face & Eye Detection** using OpenCV and Dlib/Mediapipe  
- 😴 **Drowsiness Detection** via Eye Aspect Ratio (EAR)  
- 📱 **Distraction Detection** (e.g., looking away from the road)  
- 🔊 **Automatic Alert System** — sound or visual warning when danger detected  
- 🧩 **Modular Design** — easy to integrate into other systems  
- 📊 **Performance Logging** — record detection data for analysis  

---

## 🧰 Tech Stack
| Category | Technologies |
|-----------|--------------|
| **Language** | Python |
| **Libraries** | OpenCV, NumPy, Dlib, Mediapipe, TensorFlow/Keras (optional) |
| **Tools** | Jupyter Notebook / VS Code |
| **Hardware** | Webcam or USB camera |

---

## 📁 Project Structure
    ```bash
Driver_Safety/
│
├── data/                  # Sample images or datasets
├── models/                # Trained model files (if applicable)
├── src/                   # Source code
│   ├── main.py            # Main application entry
│   ├── drowsiness.py      # Drowsiness detection logic
│   ├── alert_system.py    # Sound / warning triggers
│   └── utils.py           # Helper functions
│
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── demo_video.mp4         # Example demo video (optional)

---

## ⚙️ Installation

# Clone the repository
    ```bash
    git clone https://github.com/Dhivakar2005/Driver_Safety.git
    cd Driver_Safety


# Create a virtual environment (optional but recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # for Linux/Mac
    venv\Scripts\activate     # for Windows


# Install dependencies
    ```bash
    pip install -r requirements.txt

## 🚀 Usage

# Run the main program
    ```bash
    python app.py


  - Allow camera access when prompted.

  - The system will start monitoring the driver and issue alerts when:

  - Eyes remain closed for a certain duration (drowsiness)

  - The driver looks away from the road for too long (distraction)

## 🚧 Future Improvements

  - Add emotion detection and head pose estimation

  - Integrate with vehicle IoT systems for automatic braking

  - Support multi-driver profiles

  - Develop mobile app version for Android/iOS

## 👨‍💻 Contributors

 - Dhivakar G –  Model & Flask Developer 
 - Siva E - Frontend Developer

Contributions and pull requests are welcome!

## 📄 License

This project is licensed under the MIT License – see the LICENSE file for details.


# 💡 Safety first! This system is meant to assist drivers, not replace human responsibility.
