---
title: Raplica AI Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Raplica AI Chatbot

Hey there! This is **Raplica AI** — a custom multi-role chatbot I built that acts as a specialized assistant for different types of users: Students, Teachers, Farmers, Doctors, and Women. 

Instead of a generic AI, you can pick a role, and the bot changes its entire personality and advice style to match what you need. It also supports multiple languages (Hindi, English, Sanskrit, Tamil, Bhojpuri) and even talks back to you!

## What's under the hood?

I initially messed around with a few different setups, but right now the project is running on:
- **Backend**: Python with Flask
- **Database**: SQLite (handled via Flask-SQLAlchemy) to save users and chat histories.
- **Frontend**: Plain HTML, CSS, and Vanilla JavaScript. (I wanted to keep it super lightweight without needing React or Vue).
- **AI Brain**: Google Gemini API (`gemini-flash-lite-latest`), which is incredibly fast and generous on the free tier.
- **Voice**: It hooks directly into the browser's native Web Speech API for voice-to-text (microphone) and text-to-speech (the AI talks to you!).

## Cool Features

- **Roleplay Modes**: The AI instantly switches context depending on if you are a Farmer asking about crops or a Student needing help with homework.
- **Fully Voice Enabled**: Click the mic to speak your prompt in Hindi/English, and the AI will actually read its response out loud to you. 
- **Image & File Support**: You can upload images, PDFs, or text files, and the Gemini vision model will read them and answer your questions about them.
- **User Accounts**: I added a real sign-in and registration system. 
- **Chat History**: Just like ChatGPT, your previous conversations are saved to a local database and show up in the left sidebar.

## How to run it locally

If you want to spin this up on your own machine, it's pretty straightforward.

### 1. Get your API Key
You'll need a free Gemini API key from [Google AI Studio](https://aistudio.google.com/). 
Create a file named `.env` in the `backend/` folder and add your key:
```text
GEMINI_API_KEY=your_api_key_here
```

### 2. Install dependencies
Make sure you have Python installed. Then open your terminal, go into the `backend` folder, and run:
```bash
pip install -r requirements.txt
```

### 3. Start the server
```bash
python app.py
```
The Flask server will start up. Just open your browser and go to `http://127.0.0.1:5000`. The SQLite database will automatically create itself the first time you run it.

## License
MIT License - feel free to fork it, break it, and build something cool!
