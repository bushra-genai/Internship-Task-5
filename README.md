# Internship-Task-5

🧠 Multi-Tool Agent (Groq + File RAG + Wikipedia + Calculator)

An advanced AI assistant built with Streamlit that combines multiple tools — Groq LLM, File-based RAG, Wikipedia Search, and a Smart Calculator — into one interactive application.
This agent can answer questions, perform contextual retrieval from uploaded files, fetch real-time knowledge from Wikipedia, and handle basic mathematical computations — all in a single interface.

🚀 Features

✅ Groq Integration:
Uses Groq’s powerful API for fast and efficient LLM-based reasoning.

✅ File RAG (Retrieval-Augmented Generation):
Upload .txt, .csv, .pdf, or .docx files and let the agent extract relevant insights directly from your documents.

✅ Wikipedia Search:
Automatically fetches definitions, explanations, and facts from Wikipedia.

✅ Smart Calculator:
Handles mathematical queries seamlessly within the same chat.

✅ Streamlit UI:
Simple, modern, and responsive design with easy-to-use chat and controls.

✅ Persistent Chat Download:
Export your session with a single click using the Download Chat button.

🖼️ Interface Preview

⚙️ Installation

Clone the Repository

git clone https://github.com/yourusername/multi-tool-agent.git
cd multi-tool-agent


Create a Virtual Environment

python -m venv venv
venv\Scripts\activate    # For Windows
source venv/bin/activate # For macOS/Linux


Install Dependencies

pip install -r requirements.txt


Add Your API Key

Create a .env file in the root directory.

Add your Groq API key:

GROQ_API_KEY=your_api_key_here


Run the Application

streamlit run app.py

🧩 Usage

Enter your Groq API key in the sidebar.

Upload a knowledge file (optional).

Ask your question in the input box.

The assistant will automatically decide whether to use:

Your uploaded file (RAG)

Wikipedia

The Calculator

View the final answer below with the tool source highlighted.

🛠️ Project Structure
multi-tool-agent/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── .env                    # API key (not uploaded to GitHub)
├── README.md               # Project documentation
└── assets/
     <img width="1920" height="963" alt="Capture" src="https://github.com/user-attachments/assets/48b5d629-e449-42bd-afc0-9d1427edc7f7" /> # Screenshot

🧑‍💻 Developed By

Bushra Sarwar
💼 Machine Learning & AI Enthusiast
