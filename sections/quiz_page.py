
# sections/quiz_page.py
import streamlit as st
import json
from datetime import datetime, timezone
from genai.quiz_ai import get_quiz_hint, quiz_review_chatbot, generate_quiz
from utils.usage_db import log_usage
from utils.feedback_db import log_feedback
from utils.auth import get_user_progress, update_user_progress
from utils.db_connection import get_conn, close_conn
import logging
import plotly.express as px
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def quizzes_page(model_choice, openai_client):
    st.title("🧩 Quiz")

    # Initialize session state for quiz
    if 'current_quiz' not in st.session_state:
        st.session_state.current_quiz = None
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'quiz_chat_history' not in st.session_state:
        st.session_state.quiz_chat_history = []

    tab1, tab2 = st.tabs(["Review Quizzes", "Analytics"])

    with tab1:
        if st.session_state.current_quiz:
            display_quiz_interface(model_choice, openai_client)
        else:
            display_quiz_dashboard()

    with tab2:
        display_quiz_analytics()

def get_available_quizzes(username):
    """Get quizzes available for review (excluding completed ones)"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        # Modified query to exclude quizzes that have been attempted
        cur.execute("""
            SELECT q.quiz_id, s.name as subject, t.name as topic, 
                   t.emoji, q.quiz_data, q.created_at, 
                   COALESCE(q.status, 'ready') as status,
                   t.topic_id
            FROM quizzes q
            JOIN topics t ON q.topic_id = t.topic_id
            JOIN subjects s ON t.subject_id = s.subject_id
            WHERE q.username = %s 
            AND (q.status IS NULL OR q.status = 'ready')
            AND NOT EXISTS (
                SELECT 1 
                FROM quiz_attempts qa 
                WHERE qa.quiz_id = q.quiz_id 
                AND qa.username = %s
            )
            ORDER BY q.created_at DESC
        """, (username, username))  # Pass username twice for both conditions

        quizzes = []
        for row in cur.fetchall():
            try:
                # Handle quiz_data that might be either string or already parsed JSON
                if isinstance(row[4], str):
                    quiz_data = json.loads(row[4])
                else:
                    quiz_data = row[4]  # Already a parsed JSON object

                quiz = {
                    'id': row[0],
                    'subject': row[1],
                    'topic': row[2],
                    'emoji': row[3],
                    'questions': quiz_data,
                    'created_at': row[5],
                    'status': row[6],
                    'topic_id': row[7]
                }
                quizzes.append(quiz)
                logger.info(f"Successfully processed available quiz {quiz['id']} for topic {quiz['topic']}")
            except Exception as e:
                logger.error(f"Error processing quiz row: {str(e)}")
                continue

        logger.info(f"Retrieved {len(quizzes)} available (unattempted) quizzes")
        return quizzes
    except Exception as e:
        logger.error(f"Error fetching available quizzes: {str(e)}")
        return []
    finally:
        cur.close()
        close_conn(conn)

def get_past_quizzes(username):
    """Get completed quiz attempts"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT qa.attempt_id, s.name as subject, t.name as topic,
                   t.emoji, qa.score, qa.start_time, qa.end_time,
                   EXTRACT(EPOCH FROM (qa.end_time - qa.start_time))/60 as duration,
                   q.quiz_data
            FROM quiz_attempts qa
            JOIN quizzes q ON qa.quiz_id = q.quiz_id
            JOIN topics t ON q.topic_id = t.topic_id
            JOIN subjects s ON t.subject_id = s.subject_id
            WHERE qa.username = %s
            ORDER BY qa.end_time DESC
        """, (username,))

        attempts = []
        for row in cur.fetchall():
            try:
                # Handle quiz_data that might be either string or already parsed JSON
                quiz_data = row[8]
                if isinstance(quiz_data, str):
                    quiz_data = json.loads(quiz_data)

                attempt = {
                    'id': row[0],
                    'subject': row[1],
                    'topic': row[2],
                    'emoji': row[3],
                    'score': row[4],
                    'start_time': row[5],
                    'end_time': row[6],
                    'duration': round(row[7], 2),
                    'questions': quiz_data
                }
                attempts.append(attempt)
                logger.info(f"Successfully processed attempt {attempt['id']}")
            except Exception as e:
                logger.error(f"Error processing quiz attempt row: {str(e)}")
                continue

        logger.info(f"Retrieved {len(attempts)} past quiz attempts")
        return attempts
    except Exception as e:
        logger.error(f"Error fetching past quizzes: {str(e)}")
        return []
    finally:
        cur.close()
        close_conn(conn)

def display_quiz_dashboard():
    """Display the quiz dashboard with available and past quizzes"""
    st.subheader("📚 Available Quizzes")

    available_quizzes = get_available_quizzes(st.session_state.username)
    if available_quizzes:
        # Group quizzes by subject
        quizzes_by_subject = {}
        for quiz in available_quizzes:
            if quiz['subject'] not in quizzes_by_subject:
                quizzes_by_subject[quiz['subject']] = []
            quizzes_by_subject[quiz['subject']].append(quiz)

        # Display quizzes grouped by subject
        for subject, quizzes in quizzes_by_subject.items():
            with st.expander(f"📘 {subject}", expanded=True):
                for quiz in quizzes:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{quiz['emoji']} **{quiz['topic']}**")
                        st.write(f"Questions: {len(quiz['questions'])}")
                        st.write(f"Created: {quiz['created_at'].strftime('%d/%m/%Y %H:%M')}")
                    with col2:
                        if st.button("Start Quiz", key=f"start_{quiz['id']}", use_container_width=True):
                            st.session_state.current_quiz = quiz
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_chat_history = []
                            st.rerun()
    else:
        st.info("No quizzes available. Generate new quizzes from the study materials page.")

    st.markdown("---")
    st.subheader("📊 Past Quiz Attempts")

    past_quizzes = get_past_quizzes(st.session_state.username)
    if past_quizzes:
        for quiz in past_quizzes:
            with st.expander(f"{quiz['emoji']} {quiz['subject']} - {quiz['topic']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", f"{quiz['score']}%")
                with col2:
                    st.metric("Time Spent", f"{quiz['duration']} min")
                st.write(f"Completed: {quiz['end_time'].strftime('%d/%m/%Y %H:%M')}")

                # Show detailed review
                if st.button("Show Review", key=f"review_{quiz['id']}"):
                    st.session_state.current_quiz = {
                        'questions': quiz['questions'],
                        'attempt_id': quiz['id'],
                        'review_mode': True
                    }
                    st.rerun()
    else:
        st.info("No quiz attempts yet. Start a quiz to see your history here.")

def display_quiz_interface(model_choice, openai_client):
    """Display the quiz taking interface"""
    quiz = st.session_state.current_quiz

    # Show quiz header
    if quiz.get('review_mode'):
        st.subheader("📝 Quiz Review")
    else:
        st.subheader("📝 Quiz")

    # Exit quiz button
    if st.button("⬅️ Back to Dashboard", use_container_width=True):
        st.session_state.current_quiz = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        st.session_state.quiz_chat_history = []
        st.rerun()

    questions = quiz['questions']

    if not st.session_state.quiz_submitted:
        if 'quiz_start_time' not in st.session_state:
            st.session_state.quiz_start_time = datetime.now(timezone.utc)

        # Display questions
        for i, question in enumerate(questions, 1):
            st.write(f"**Question {i}:** {question['question']}")

            if question.get('type') == 'multiple-choice':
                st.session_state.quiz_answers[i] = st.radio(
                    f"Select your answer for Question {i}",
                    question['options'],
                    key=f"q_{i}"
                )
            else:
                st.session_state.quiz_answers[i] = st.text_input(
                    f"Your answer for Question {i}",
                    key=f"q_{i}"
                )

            # Hint feature
            with st.expander("💡 Get a Hint", expanded=False):
                if st.button(f"Show Hint", key=f"hint_{i}"):
                    try:
                        hint, usage = get_quiz_hint(
                            question,
                            st.session_state.rag_pipeline,
                            model_choice,
                            openai_client,
                            st.session_state.subject,
                            st.session_state.class_year,
                            st.session_state.language,
                            st.session_state.difficulty
                        )
                        st.info(f"Hint: {hint}")
                        st.session_state.token_tracker.update(usage, model_choice)
                        # Track hint usage
                        track_hint_usage(i)
                    except Exception as e:
                        st.error(f"Error getting hint: {str(e)}")

            st.markdown("---")

        # Submit button
        if st.button("📤 Submit Quiz", use_container_width=True):
            end_time = datetime.now(timezone.utc)

            # Calculate initial score for multiple choice questions
            correct_answers = 0
            total_questions = len(questions)

            for i, question in enumerate(questions, 1):
                if question.get('type') == 'multiple-choice':
                    user_answer = st.session_state.quiz_answers.get(i)
                    correct_answer = question.get('correct_answer')
                    if user_answer == correct_answer:
                        correct_answers += 1

            # Calculate preliminary score (will be refined in display_quiz_results)
            initial_score = (correct_answers / total_questions) * 100

            # Save attempt with initial score
            save_quiz_attempt(
                quiz_id=quiz['id'],
                start_time=st.session_state.quiz_start_time,
                end_time=end_time,
                score=initial_score,
                user_answers=st.session_state.quiz_answers
            )

            st.session_state.quiz_submitted = True
            st.rerun()

    else:
        display_quiz_results(model_choice, openai_client)

def start_quiz(quiz):
    """Initialize a new quiz attempt"""
    st.session_state.current_quiz = quiz  # This should include the 'id' field
    st.session_state.quiz_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.hints_used = set()
    st.session_state.quiz_start_time = datetime.now(timezone.utc)
    st.session_state.attempt_saved = False
    logger.info(f"Starting quiz {quiz.get('id')} at {st.session_state.quiz_start_time}")

def display_quiz_results(model_choice, openai_client):
    # Add debugging at the start
    logger.info("Starting display_quiz_results")
    logger.info(f"Session state keys: {st.session_state.keys()}")

    quiz = st.session_state.current_quiz
    questions = quiz['questions']
    correct_answers = 0
    total_score = 0
    open_answer_scores = {}
    hint_used = st.session_state.get('hints_used', set())
    
    # Log the quiz data
    logger.info(f"Current quiz: {quiz.get('id')}")
    logger.info(f"User answers: {st.session_state.quiz_answers}")
    
    start_time = st.session_state.get('quiz_start_time', datetime.now(timezone.utc))
    end_time = datetime.now(timezone.utc)

    # Process all questions and calculate score first
    for i, question in enumerate(questions, 1):
        with st.expander(f"Question {i} Review", expanded=True):
            st.write(f"**Question:** {question['question']}")
            user_answer = st.session_state.quiz_answers.get(i, 'Not answered')
            st.write(f"**Your Answer:** {user_answer}")
            
            correct_answer = question.get('correct_answer') or question.get('answer')
            
            if question.get('type') == 'multiple-choice':
                if user_answer == correct_answer:
                    st.success("✅ Correct!")
                    correct_answers += 1
                else:
                    st.error("❌ Incorrect")
                st.write(f"**Correct Answer:** {correct_answer}")
            else:
                # Evaluate open-ended answer
                score, feedback, usage = evaluate_open_answer(
                    question['question'],
                    user_answer,
                    correct_answer,
                    model_choice,
                    openai_client,
                    st.session_state.subject,
                    st.session_state.class_year,
                    st.session_state.language,
                    st.session_state.difficulty
                )
                open_answer_scores[i] = score
                st.write(f"**Score:** {score}/100")
                st.write(f"**Feedback:** {feedback}")
                st.write(f"**Model Answer:** {correct_answer}")
                
                if usage:
                    st.session_state.token_tracker.update(usage, model_choice)

    # Calculate final score before saving attempt
    final_score = calculate_quiz_score(
        questions, 
        st.session_state.quiz_answers,
        hint_used,
        open_answer_scores
    )
    
    # Log before saving attempt
    logger.info(f"Attempting to save quiz with score {final_score}")
    logger.info(f"Quiz answers to save: {st.session_state.quiz_answers}")

    # Save the attempt if not already saved
    if not st.session_state.get('attempt_saved', False):
        quiz_id = quiz.get('id')
        if quiz_id:
            # Explicitly pass all required arguments
            attempt_id = save_quiz_attempt(
                quiz_id=quiz_id,
                start_time=start_time,
                end_time=end_time,
                score=final_score,
                user_answers=st.session_state.quiz_answers
            )
            if attempt_id:
                st.session_state.attempt_saved = True
                logger.info(f"Quiz attempt {attempt_id} saved successfully")
            else:
                st.error("Failed to save quiz attempt")
        else:
            logger.error("Missing quiz ID")
            st.error("Could not save quiz attempt - missing quiz ID")

    st.markdown("### 📊 Detailed Score Breakdown")

    # Create columns for score components
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Multiple Choice Questions")
        mc_questions = len([q for q in questions if q.get('type') == 'multiple-choice'])
        if mc_questions > 0:
            mc_correct = sum(1 for i, q in enumerate(questions, 1) 
                           if q.get('type') == 'multiple-choice' 
                           and st.session_state.quiz_answers.get(i) == q.get('correct_answer'))  # Changed here
            mc_score = (mc_correct / mc_questions) * 100
            st.write(f"✓ Correct answers: {mc_correct}/{mc_questions}")
            st.write(f"📊 Score: {mc_score:.1f}%")
        else:
            st.write("No multiple choice questions in this quiz")

    with col2:
        st.markdown("#### Open-Ended Questions")
        open_questions = len([q for q in questions if q.get('type') == 'open-ended'])
        if open_questions > 0:
            avg_open_score = sum(open_answer_scores.values()) / open_questions
            st.write(f"📝 Questions evaluated: {open_questions}")
            st.write(f"📊 Average score: {avg_open_score:.1f}%")
        else:
            st.write("No open-ended questions in this quiz")

    st.markdown("---")
    st.markdown("#### 🎯 Score Calculation")

    # Explain the weighting if both types of questions exist
    if mc_questions > 0 and open_questions > 0:
        st.write("""
        The final score is calculated with the following weights:
        - Multiple Choice Questions: 60%
        - Open-Ended Questions: 40%
        """)

        weighted_mc = mc_score * 0.6 if mc_questions > 0 else 0
        weighted_open = avg_open_score * 0.4 if open_questions > 0 else 0
        base_score = weighted_mc + weighted_open
    else:
        # If only one type of question exists
        base_score = mc_score if mc_questions > 0 else avg_open_score
        st.write("Final score is based on 100% weight for the single question type present.")

    # Display hint penalties if any were used
    if hint_used:
        st.markdown("#### 💡 Hint Penalties")
        penalty_per_hint = 5
        total_penalty = len(hint_used) * penalty_per_hint
        st.write(f"""
        - Number of hints used: {len(hint_used)}
        - Penalty per hint: -{penalty_per_hint}%
        - Total penalty: -{total_penalty}%
        """)

    # Final Score Display with explanation
    st.markdown("#### 🏆 Final Score")
    st.markdown(f"""
    Base Score: **{base_score:.1f}%**
    {'- Hint Penalty: **-' + str(len(hint_used) * 5) + '%**' if hint_used else ''}

    **Final Score: {final_score:.1f}%**
    """)

    # Performance assessment
    st.markdown("#### 📈 Performance Assessment")
    if final_score == 100:
        st.success("🌟 Perfect Score! Outstanding performance across all questions!")
    elif final_score >= 90:
        st.success("🎯 Excellent! Near-perfect understanding of the material!")
    elif final_score >= 80:
        st.success("👏 Great work! You've demonstrated strong knowledge!")
    elif final_score >= 70:
        st.info("📚 Good job! Some areas for review, but overall solid understanding.")
    elif final_score >= 60:
        st.info("📝 Fair performance. Consider reviewing the topics you missed.")
    else:
        st.warning("💪 This topic might need more review. Don't give up!")

    # Add tips for improvement if score is less than perfect
    if final_score < 100:
        st.markdown("#### 💡 Tips for Improvement")
        tips = []
        # Use mc_score only if it's defined (i.e., if there are MC questions)
        if mc_questions > 0 and ('mc_score' in locals() and mc_score < 90):
            tips.append("- Review the multiple choice questions you missed, focusing on understanding why the correct answer was right.")
        if open_questions > 0 and ('avg_open_score' in locals() and avg_open_score < 90):
            tips.append("- For open-ended questions, practice writing more complete and detailed answers.")
        if hint_used:
            tips.append("- Try to solve questions without hints to improve your score next time.")
        if tips:
            st.write("\n".join(tips))

    if st.session_state.get('attempt_saved', False):
        if st.button("Return to Quiz Dashboard"):
            st.session_state.current_quiz = None
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.hints_used = set()
            st.rerun()
    
    # Review chat interface
    st.markdown("---")
    st.subheader("💭 Review Chat")

    # Display chat history
    for message in st.session_state.quiz_chat_history:
        if message["role"] == "user":
            st.markdown(f"🙋‍♂️ **You:** {message['content']}")
        else:
            st.markdown(f"🤖 **Tutor:** {message['content']}")

            # Feedback for AI responses
            with st.expander("📢 Feedback", expanded=False):
                display_chat_feedback(message['content'], model_choice)

    # Chat input
    user_input = st.text_input("Ask a question about the quiz:", key="quiz_review_input")
    if st.button("Send", key="quiz_review_send"):
        if user_input:
            with st.spinner("Getting response..."):
                try:
                    response, usage = quiz_review_chatbot(
                        st,
                        st.session_state.rag_pipeline,
                        model_choice,
                        openai_client,
                        questions,
                        st.session_state.quiz_answers,
                        st.session_state.subject,
                        st.session_state.class_year,
                        st.session_state.language,
                        st.session_state.difficulty,
                        user_input
                    )

                    st.session_state.quiz_chat_history.append(
                        {"role": "user", "content": user_input}
                    )
                    st.session_state.quiz_chat_history.append(
                        {"role": "assistant", "content": response}
                    )

                    st.session_state.token_tracker.update(usage, model_choice)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error in chat: {str(e)}")
        else:
            st.warning("Please enter a question.")

    if st.button("🔄 Reset Chat"):
        st.session_state.quiz_chat_history = []
        st.rerun()

def calculate_quiz_score(quiz_data, user_answers, hint_used, open_answer_scores):
    """
    Calculates final quiz score considering hints and open answer evaluations
    """
    total_questions = len(quiz_data)
    total_possible_score = 100
    hint_penalty = 5  # 5% penalty per hint used

    # Calculate base score for multiple choice questions
    mc_correct = 0
    mc_total = 0
    open_total = 0

    for i, question in enumerate(quiz_data, 1):
        if question.get('type') == 'multiple-choice':
            mc_total += 1
            if user_answers.get(i) == question.get('correct_answer'):
                mc_correct += 1
        else:
            open_total += 1

    # Calculate weighted scores
    mc_weight = 0.6 if open_total > 0 else 1.0  # 60% weight if there are open questions
    open_weight = 0.4 if mc_total > 0 else 1.0   # 40% weight if there are MC questions

    # Calculate multiple choice score
    mc_score = (mc_correct / mc_total * 100) if mc_total > 0 else 0

    # Calculate open answer score
    open_score = (sum(open_answer_scores.values()) / open_total) if open_total > 0 else 0

    # Calculate final weighted score
    final_score = (mc_score * mc_weight) + (open_score * open_weight)

    # Apply hint penalties
    total_penalty = len(hint_used) * hint_penalty
    final_score = max(0, final_score - total_penalty)

    return round(final_score, 2)

def evaluate_open_answer(question, user_answer, correct_answer, model_choice, openai_client, subject, class_year, language, difficulty):
    """
    Evaluates an open-ended answer using GPT-4o mini
    Returns:
        tuple: (score, feedback, usage statistics)
    """
    messages = [
        {
            "role": "system",
            "content": (
                f"Sei un docente esperto di {subject} per studenti del {class_year}º anno delle scuole superiori "
                f"in Italia di livello {difficulty}. Il tuo compito è valutare le risposte degli studenti fornendo "
                f"un punteggio da 0 a 100 e un feedback costruttivo e dettagliato. Considera l'accuratezza, "
                f"la completezza e la chiarezza nella tua valutazione. Rispondi in {language} con un oggetto JSON "
                f"contenente i campi 'score' e 'feedback'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Valuta questa risposta dello studente:\n\n"
                f"Domanda: {question}\n"
                f"Risposta modello: {correct_answer}\n"
                f"Risposta dello studente: {user_answer}\n\n"
                "Fornisci un punteggio (0-100) e un feedback specifico che spieghi la valutazione."
            )
        }
    ]

    try:
        response = openai_client.chat.completions.create(
            model=model_choice,
            messages=messages,
            response_format={ "type": "json_object" },
            max_tokens=300,
            temperature=0.3,
        )

        evaluation = json.loads(response.choices[0].message.content)

        # Extract usage statistics
        usage = {
            'total_tokens': response.usage.total_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
        }

        score = evaluation.get('score', 0)
        feedback = evaluation.get('feedback', "Errore nella valutazione")

        return score, feedback, usage

    except Exception as e:
        logger.error(f"Errore nella valutazione della risposta: {str(e)}")
        return 0, "Si è verificato un errore durante la valutazione.", {}

def track_hint_usage(question_number):
    """Track when hints are used"""
    if 'hints_used' not in st.session_state:
        st.session_state.hints_used = set()
    st.session_state.hints_used.add(question_number)
    
def display_quiz_analytics():
    """Display comprehensive quiz analytics"""
    st.title("📊 Quiz Analytics")

    # Overall Statistics
    with st.expander("📈 Overall Performance", expanded=True):
        stats = get_quiz_statistics(st.session_state.username)
        if stats and stats['total_quizzes'] > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Quizzes Taken", stats['total_quizzes'])
            with col2:
                st.metric("Average Score", f"{stats['avg_score']:.1f}%")
            with col3:
                st.metric("Average Time", f"{stats['avg_time']:.1f} min")
        else:
            st.info("No quiz attempts yet. Complete some quizzes to see your statistics.")

    # Performance by Subject
    with st.expander("📚 Performance by Subject", expanded=True):
        subject_stats = get_subject_performance(st.session_state.username)
        if not subject_stats.empty:
            fig = px.bar(
                subject_stats,
                x='subject',
                y='avg_score',
                title='Average Score by Subject',
                labels={'subject': 'Subject', 'avg_score': 'Average Score (%)'}
            )
            st.plotly_chart(fig)
        else:
            st.info("Complete quizzes in different subjects to see performance comparison.")

    # Progress Over Time
    with st.expander("📈 Progress Over Time", expanded=True):
        progress_data = get_progress_over_time(st.session_state.username)
        if not progress_data.empty:
            fig = px.line(
                progress_data,
                x='date',
                y='score',
                title='Quiz Scores Over Time',
                labels={'date': 'Date', 'score': 'Score (%)'}
            )
            st.plotly_chart(fig)
        else:
            st.info("Complete more quizzes to see your progress over time.")

def get_quiz_statistics(username):
    """Get overall quiz statistics"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                COUNT(*) as total_quizzes,
                AVG(score) as avg_score,
                AVG(EXTRACT(EPOCH FROM (end_time - start_time))/60) as avg_time
            FROM quiz_attempts
            WHERE username = %s
        """, (username,))

        result = cur.fetchone()
        if result:
            return {
                'total_quizzes': result[0],
                'avg_score': result[1] or 0,
                'avg_time': result[2] or 0
            }
        return None
    except Exception as e:
        logger.error(f"Error getting quiz statistics: {e}")
        return None
    finally:
        cur.close()
        close_conn(conn)

def get_subject_performance(username):
    """Get quiz performance statistics by subject"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                s.name as subject,
                COUNT(*) as attempts,
                AVG(qa.score) as avg_score,
                AVG(EXTRACT(EPOCH FROM (qa.end_time - qa.start_time))/60) as avg_time
            FROM quiz_attempts qa
            JOIN quizzes q ON qa.quiz_id = q.quiz_id
            JOIN topics t ON q.topic_id = t.topic_id
            JOIN subjects s ON t.subject_id = s.subject_id
            WHERE qa.username = %s
            GROUP BY s.name
            ORDER BY avg_score DESC
        """, (username,))

        results = []
        for row in cur.fetchall():
            results.append({
                'subject': row[0],
                'attempts': row[1],
                'avg_score': round(row[2], 2) if row[2] else 0,
                'avg_time': round(row[3], 2) if row[3] else 0
            })
        return pd.DataFrame(results)
    except Exception as e:
        logger.error(f"Error getting subject performance: {e}")
        return pd.DataFrame()
    finally:
        cur.close()
        close_conn(conn)

def get_progress_over_time(username):
    """Get quiz performance progress over time"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                DATE(end_time) as quiz_date,
                AVG(score) as avg_score,
                COUNT(*) as quizzes_taken
            FROM quiz_attempts
            WHERE username = %s
            GROUP BY DATE(end_time)
            ORDER BY quiz_date
        """, (username,))

        results = []
        for row in cur.fetchall():
            results.append({
                'date': row[0],
                'score': round(row[1], 2) if row[1] else 0,
                'count': row[2]
            })
        return pd.DataFrame(results)
    except Exception as e:
        logger.error(f"Error getting progress data: {e}")
        return pd.DataFrame()
    finally:
        cur.close()
        close_conn(conn)

def save_quiz_attempt(quiz_id, start_time, end_time, score, user_answers):
    """
    Save a quiz attempt to the database
    """
    logger.info(f"save_quiz_attempt called with: quiz_id={quiz_id}, score={score}, answers_count={len(user_answers)}")

    conn = get_conn()
    cur = conn.cursor()
    try:
        # Ensure answers are stored as JSON string
        answers_json = json.dumps(user_answers)

        # First insert the attempt
        cur.execute("""
            INSERT INTO quiz_attempts (
                quiz_id, 
                username, 
                score, 
                start_time, 
                end_time, 
                answers,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING attempt_id
        """, (
            quiz_id,
            st.session_state.username,
            score,
            start_time,
            end_time,
            answers_json,
            datetime.now(timezone.utc)
        ))

        attempt_id = cur.fetchone()[0]
        logger.info(f"Created quiz attempt {attempt_id}")

        # Update quiz status to 'completed'
        cur.execute("""
            UPDATE quizzes
            SET status = 'completed',
                last_attempt = %s,
                completion_rate = (
                    SELECT COUNT(*) * 100.0 / NULLIF((SELECT COUNT(*) FROM quiz_attempts WHERE quiz_id = %s), 0)
                    FROM quiz_attempts 
                    WHERE quiz_id = %s AND score >= 60
                ),
                average_score = (
                    SELECT AVG(score)
                    FROM quiz_attempts
                    WHERE quiz_id = %s
                )
            WHERE quiz_id = %s
        """, (end_time, quiz_id, quiz_id, quiz_id, quiz_id))

        conn.commit()
        return attempt_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving quiz attempt: {str(e)}")
        raise
    finally:
        cur.close()
        close_conn(conn)

def display_chat_feedback(response_content, model_choice):
    """Display feedback interface for chat responses"""
    cols = st.columns([1, 1, 4])
    feedback_id = f"feedback_{hash(response_content)}"

    if feedback_id not in st.session_state:
        st.session_state[feedback_id] = {'rating': None, 'comment': ''}

    with cols[0]:
        if st.button("👍", key=f"thumbs_up_{feedback_id}"):
            st.session_state[feedback_id]['rating'] = 1
    with cols[1]:
        if st.button("👎", key=f"thumbs_down_{feedback_id}"):
            st.session_state[feedback_id]['rating'] = 0

    st.session_state[feedback_id]['comment'] = st.text_input(
        "Additional comments (optional)",
        key=f"comment_{feedback_id}"
    )

    if st.button("Submit Feedback", key=f"submit_{feedback_id}"):
        if st.session_state[feedback_id]['rating'] is not None:
            log_feedback(
                st.session_state.username,
                'quiz_chat',
                feedback_id,
                '',
                response_content,
                st.session_state[feedback_id]['rating'],
                st.session_state[feedback_id]['comment'],
                model_choice,
                json.dumps(st.session_state.quiz_chat_history)
            )
            st.success("Thank you for your feedback!")
            st.session_state[feedback_id] = {'rating': None, 'comment': ''}
            st.rerun()
        else:
            st.warning("Please select a rating (👍 or 👎) before submitting.")

def update_user_progress_quiz(username):
    """Update user progress after completing a quiz"""
    try:
        current_progress = get_user_progress(username)
        current_progress['quizzes_taken'] = current_progress.get('quizzes_taken', 0) + 1
        update_user_progress(username, current_progress)
    except Exception as e:
        logger.error(f"Error updating user progress: {e}")