
# AI-Powered High School Tutor Platform

## Overview
An intelligent tutoring system designed specifically for Italian high school students, leveraging AI to provide personalized learning experiences based on uploaded study materials.

## Key Features
- **Smart Diary**: student can write their assignments and test dates which are then parsed and shown in the activities
- **Daily Activities**: comprehensive daily activity dashboard with upcoming tests, assigments and flashcard review reminders
- **Study Materials Upload**: student can upload their study materials under their topic and subject of interest
- **AI Tutor**: Interactive chat-based tutoring with context-aware responses based on uploaded materials
- **Smart Flashcards**: AI-generated flashcards with spaced repetition implementation
- **Personalized Quizzes**: Mix of multiple choice and open ended question with personalized feedback and scoring

## Technical Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **AI Integration**: OpenAI
- **Data Storage**: SQLite
- **RAG Pipeline**: Custom implementation for context-aware responses

## Smart Features
- RAG (Retrieval Augmented Generation) for context-aware responses
- Spaced repetition algorithms for optimal learning
- Token usage tracking and optimization
- Feedback collection and analysis
- Administrative dashboard for system monitoring

## Installation and Setup
1. Clone the repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Configure OpenAI API credentials
4. Run the application: `streamlit run main.py`
