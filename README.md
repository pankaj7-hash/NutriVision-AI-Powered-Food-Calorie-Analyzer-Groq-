🥗 NutriVision — AI-Powered Food & Calorie Analyzer (Groq Vision)

NutriVision is an AI computer vision application that analyzes meal photos and estimates calories for each food item and total calorie intake using Groq Vision (Llama-4 models).
Built with Streamlit, this tool provides fast, interactive, and intelligent nutrition analysis.

Developed by Dr. Pankaj Mahure

🚀 Features

📷 Upload meal images (JPG, JPEG, PNG)

🍛 Detect visible food items automatically

🔥 Estimate calories per item

➕ Calculate total calories

⚡ Powered by Groq Vision API (Llama-4 Scout / Maverick)

🎛 Adjustable AI creativity (temperature)

🖥 Clean Streamlit interface

⏱ Fast real-time AI inference

📸 App Screenshot
<img width="1865" height="907" alt="NutriVision app_screenshot" src="https://github.com/user-attachments/assets/9da2b7e9-a3f0-40c9-a09a-93eb894f70ab" />

🧠 How It Works
User uploads food image
        │
        ▼
Streamlit Web Interface
        │
        ▼
Image converted to Base64
        │
        ▼
Groq Vision API (Llama-4 Vision Model)
        │
        ▼
AI detects food items & estimates calories
        │
        ▼
Results displayed to user
📂 Project Structure
nutrivision-groq/
│
├── app.py
├── requirements.txt
├── .env
├── README.md
└── screenshots/
    └── nutrivision.png
⚙️ Requirements

Install dependencies:

pip install -r requirements.txt

requirements.txt

streamlit
pillow
requests
python-dotenv
🔑 Setup Groq API Key

Get API key from:
https://console.groq.com/keys

Create .env file:

GROQ_API_KEY=your_api_key_here
▶️ Run the Application
streamlit run app.py

App will open at:

http://localhost:8501
🧪 Example

Upload image containing:

Rice

Dal

Vegetables

Example output:

1) Rice — ~200 calories
2) Dal — ~150 calories
3) Vegetables — ~80 calories

Total — ~430 calories
🧩 Tech Stack

Python

Streamlit

Groq API

Llama-4 Vision Models

Pillow

Requests

dotenv

🎯 Use Cases

Personal nutrition tracking

Diet planning

Healthcare monitoring

Fitness applications

Public health research

AI computer vision projects

🔒 Disclaimer

Calorie estimates are approximate and may vary based on portion size and preparation method.

👨‍💻 Author

Dr. Pankaj Mahure
Public Health Professional | AI Developer | Data Scientist

⭐ Support

If you like this project:

Star ⭐ the repository
Fork 🍴 the repository
Contribute 👨‍💻

📌 Future Improvements

Automatic portion size estimation

Nutrition breakdown (protein, carbs, fats)

Multiple image support

Mobile deployment

Cloud deployment
