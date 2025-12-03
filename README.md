 (REAL-ESTATE-BACKEND)


🏠 Real Estate AI Chatbot – Backend

This is the backend for the Real Estate AI Chatbot system.  
It is developed using Django + Django REST Framework and deployed on Render.

The backend:
- Receives user queries
- Processes them using Groq LLM
- Generates summaries, charts & tables
- Produces PDF reports
- Sends structured responses to the frontend

---

 🚀 Tech Stack

- Python
- Django
- Django REST Framework
- Groq LLM (AI Integration)
- SQLite
- Render (Deployment)

---

🌐 Live Backend API

Deployed on Render:  
👉 https://realestateagent-ol6i.onrender.com

---

🔌 API Endpoints

| Method | Endpoint           | Description               |
|--------|--------------------|---------------------------|
| POST   | /api/analyze/      | Analyze user query        |
| POST   | /api/download-pdf/ | Generate and download PDF |

---

✅ Complete Local Setup Instructions

Follow these steps exactly in order:

1️⃣ Clone the repository
git clone (https://github.com/urvashi-lab/RealEstateAgent.git)
cd backend

2️⃣ Create & activate virtual environment

Windows

python -m venv venv
venv\Scripts\activate


Mac/Linux

python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Create .env file (IMPORTANT)

Create a .env file inside the backend root folder:

SECRET_KEY=your_django_secret_key
OPENAI_API_KEY=your_groq_api_key
DEBUG=True


⚠️ Do NOT upload .env to GitHub.

5️⃣ Run database migrations
python manage.py makemigrations
python manage.py migrate

6️⃣ Create admin user (optional)
python manage.py createsuperuser

7️⃣ Start the development server
python manage.py runserver


Server will start at:

http://127.0.0.1:8000/

🔗 Frontend Connection

Your frontend must point to:

https://realestateagent-ol6i.onrender.com/api/analyze/

https://realestateagent-ol6i.onrender.com/api/download-pdf/

🤖 AI Integration (Groq LLM)

Uses Groq API as LLM engine

Processes natural language queries

Converts AI results into:

Summary text

Tabular data

Charts metadata

PDF report

📄 PDF Generation Flow

Frontend sends data to /api/download-pdf/

Backend generates PDF using Python

File is streamed back as a download

Frontend triggers auto-download

☁️ Deployment on Render
Render Build Command:
pip install -r requirements.txt

Start Command:
gunicorn config.wsgi:application

Environment Variables on Render:
SECRET_KEY=xxxx
GROQ=xxxx
DEBUG=False

🛡️ Security Notes

DEBUG must be False in production

SECRET_KEY must never be committed

CORS is enabled for frontend integration

👩‍💻 Author

Urvashi Patil
Electronics & Computer Engineering
Full Stack Developer | Machine Learning Enthusiast

⚠️ Disclaimer

This project is for academic and demonstration purposes only.
