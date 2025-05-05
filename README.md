# HiFi Image Forgery Detection

This project is designed for detecting image forgeries using the HiFi_Net model. It uses Python 3.7.16 and a Conda virtual environment.

## 📦 Requirements

- Python 3.7.16
- Conda
- `requirements.txt` file with all necessary dependencies

## ⚙️ Setup Instructions

Follow the steps below to set up and run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/siddhartha1104/imageForgeryDetection.git
cd imageForgeryDetection
```

### 2. Create Virtual Environment

```bash
conda create -p ./venv python=3.7.16 -c conda-forge -y
```

### 3. Activate the Environment

```bash
conda activate ./venv
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Run the Project

To run the main script:

```bash
python HiFi_Net.py
```

## 📌 Notes

- Make sure to activate the virtual environment every time before running the script.
- The `requirements.txt` file should include all project dependencies.
- Use the exact Python version for compatibility.

## 📁 Project Structure (Optional)

```
HiFi_IFDL/
├── venv/                   # Conda virtual environment
├── HiFi_Net.py             # Main script
├── requirements.txt        # List of dependencies
├── README.md               # Project setup and usage instructions
└── ...                     # Other project files and modules
```

## 🛠️ Troubleshooting

- If `conda` is not recognized, ensure it's installed and added to your system PATH.
- If you face package compatibility issues, consider recreating the environment from scratch.
