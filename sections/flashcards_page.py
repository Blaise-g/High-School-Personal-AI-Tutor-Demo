# sections/flashcards_review.py
import streamlit as st
from genai.flash_ai import flash_chat, explain_flashcard, provide_example, create_mnemonic
from utils.usage_db import log_usage
from utils.feedback_db import log_feedback
from utils.auth import get_user_progress, update_user_progress
from genai.rag_pipeline import retrieve_relevant_docs
from utils.fsrs import fsrs_manager
from datetime import datetime, timezone
import json
from utils.db_connection import get_conn, close_conn
import logging
import pandas as pd
import plotly.express as px

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def flashcards_page(model_choice, openai_client):
    logger.info("Entering flashcards page")
    st.title("📝 Flashcards")

    # Initialize session state variables
    if 'current_flashcard_index' not in st.session_state:
        st.session_state.current_flashcard_index = 0
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'show_answer' not in st.session_state:
        st.session_state.show_answer = False
    if 'reviewed_flashcards' not in st.session_state:
        st.session_state.reviewed_flashcards = []
    if 'session_complete' not in st.session_state:
        st.session_state.session_complete = False
    if 'current_deck' not in st.session_state:
        st.session_state.current_deck = None

    tab1, tab2 = st.tabs(["Review Decks", "Analytics"])

    with tab1:
        if not st.session_state.current_deck:
            display_deck_selection()
        else:
            display_review_interface(model_choice, openai_client)

    with tab2:
        display_review_statistics(st)

def display_deck_selection():
    """Display available flashcard decks for review"""
    # Clear session state for new selection
    if 'current_session_cards' in st.session_state:
        del st.session_state.current_session_cards
    if 'total_deck_cards' in st.session_state:
        del st.session_state.total_deck_cards
    if 'pending_reviews' in st.session_state:
        del st.session_state.pending_reviews

    st.subheader("📚 Available Decks")
    available_decks = get_available_decks(st.session_state.username)

    if not available_decks:
        display_upcoming_reviews()
        return

    # Group decks by subject
    decks_by_subject = {}
    for subject, topic, total_cards, reviewed_cards, next_review, topic_id, due_now, upcoming_times in available_decks:
        if subject not in decks_by_subject:
            decks_by_subject[subject] = []

        upcoming = [datetime.fromisoformat(t) for t in (upcoming_times.split(',') if upcoming_times else [])]
        upcoming = [t for t in upcoming if t > datetime.now(timezone.utc)]

        decks_by_subject[subject].append({
            'topic': topic,
            'total_cards': total_cards,
            'reviewed_cards': reviewed_cards or 0,
            'due_now': due_now,
            'next_review': next_review,
            'topic_id': topic_id,
            'upcoming_reviews': upcoming
        })

    # Display decks grouped by subject
    for subject, decks in decks_by_subject.items():
        with st.expander(f"📘 {subject}", expanded=True):
            for deck in decks:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Topic:** {deck['topic']}")
                    if deck['due_now'] > 0:
                        st.write(f"**Cards due now:** {deck['due_now']}")

                    if deck['upcoming_reviews']:
                        next_times = [t.strftime('%H:%M') for t in deck['upcoming_reviews']]
                        st.write(f"**Upcoming today at:** {', '.join(next_times)}")

                with col2:
                    if deck['due_now'] > 0:
                        if st.button("Review", key=f"review_{subject}_{deck['topic']}"):
                            start_review_session({
                                'subject': subject,
                                'topic': deck['topic'],
                                'topic_id': deck['topic_id']
                            })
                            st.rerun()

def display_upcoming_reviews():
    """Display information about upcoming reviews when no cards are currently due"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        now = datetime.now(timezone.utc)
        logger.info(f"Checking upcoming reviews at {now}")

        cur.execute("""
            WITH upcoming AS (
                SELECT 
                    next_review,
                    COUNT(*) as card_count,
                    MIN(flashcard_id) as sample_id
                FROM flashcards
                WHERE username = %s
                AND next_review > CURRENT_TIMESTAMP
                AND NOT reviewed_today
                AND DATE(next_review) = CURRENT_DATE
                GROUP BY next_review
                ORDER BY next_review
                LIMIT 3
            )
            SELECT next_review, card_count, sample_id
            FROM upcoming
        """, (st.session_state.username,))

        upcoming_reviews = cur.fetchall()
        logger.info(f"Found {len(upcoming_reviews)} upcoming review groups")

        for review in upcoming_reviews:
            logger.info(f"Review group: {review[0]}, Count: {review[1]}, Sample ID: {review[2]}")

        if not upcoming_reviews:
            st.info("No more reviews scheduled for today.")
            return

        st.info("✅ All currently due flashcards have been reviewed!")

        st.markdown("### 📅 Upcoming Reviews Today")
        for next_review, count, _ in upcoming_reviews:
            st.write(f"• {count} cards at {next_review.strftime('%H:%M')}")

    finally:
        cur.close()
        close_conn(conn)

def start_review_session(deck):
    """Initialize a new review session"""
    logger.info(f"Starting review session for deck: {deck}")

    st.session_state.current_deck = deck
    st.session_state.current_flashcard_index = 0
    st.session_state.conversation_history = []
    st.session_state.show_answer = False
    st.session_state.reviewed_flashcards = []
    st.session_state.session_complete = False

    # Get all cards for the session at once
    all_cards = fsrs_manager.get_next_flashcards(st.session_state.username)
    filtered_cards = [
        card for card in all_cards 
        if card.get('subject_name', '').lstrip('#') == deck['subject'] 
        and card.get('topic_name', '') == deck['topic']
    ]

    # Store the cards and count in session state
    st.session_state.current_session_cards = filtered_cards
    st.session_state.total_deck_cards = len(filtered_cards)
    st.session_state.pending_reviews = []

    logger.info(f"Started review session with {len(filtered_cards)} cards")

def display_review_interface(model_choice, openai_client):
    """Display the flashcard review interface for the selected deck"""
    # Use stored session cards instead of fetching new ones
    if 'current_session_cards' not in st.session_state:
        all_flashcards = fsrs_manager.get_next_flashcards(st.session_state.username)
        flashcards = get_filtered_flashcards(all_flashcards)
        st.session_state.current_session_cards = flashcards
    else:
        flashcards = st.session_state.current_session_cards

    if not flashcards:
        display_no_flashcards_message(None)
        return

    # Display current deck info
    st.markdown(f"**Current Deck:** {st.session_state.current_deck['subject']} - {st.session_state.current_deck['topic']}")

    # If session is complete, show completion message
    if st.session_state.session_complete:
        display_session_complete()
        return

    # Display flashcards
    display_flashcard_content(st, model_choice, openai_client, flashcards)

def display_session_complete():
    """Display completion message and options"""
    st.success("🎉 Congratulations! You've completed this deck!")
    st.balloons()

    st.markdown("""
    ### What would you like to do next?
    """)

    if st.button("📚 Review Another Deck"):
        st.session_state.current_deck = None
        st.session_state.session_complete = False
        st.session_state.current_flashcard_index = 0
        st.session_state.reviewed_flashcards = []
        st.rerun()


def display_flashcard_content(st, model_choice, openai_client, flashcards):
    """Display the current flashcard with all learning features"""
    try:
        if not flashcards:
            st.warning("No flashcards available for review")
            return

        current_index = st.session_state.current_flashcard_index
        total_cards = st.session_state.total_deck_cards

        logger.info(f"Displaying card {current_index + 1} of {total_cards}")

        # Show deck progress
        st.markdown(f"### Deck Progress: {current_index + 1}/{total_cards}")
        progress = (current_index + 1) / total_cards
        st.progress(progress)

        current_card = flashcards[current_index]

        col_card, col_approfondimento = st.columns([1, 1])

        with col_card:
            st.markdown("### Flashcard")

            # Navigation and progress
            col_prev, col_progress, col_next = st.columns([1, 2, 1])
            with col_prev:
                if current_index > 0 and st.button("⬅️ Previous"):
                    st.session_state.current_flashcard_index -= 1
                    st.session_state.conversation_history = []
                    st.session_state.show_answer = False
                    st.rerun()

            with col_progress:
                st.markdown(f"**Progress:** {current_index + 1}/{len(flashcards)}")
                progress = (current_index + 1) / len(flashcards)
                st.progress(progress)

            with col_next:
                if current_index < len(flashcards) - 1 and st.button("Next ➡️"):
                    st.session_state.current_flashcard_index += 1
                    st.session_state.conversation_history = []
                    st.session_state.show_answer = False
                    st.rerun()

            # Flashcard content
            with st.container():
                st.markdown("#### Front")
                st.info(current_card['question'])

            show_answer = st.checkbox("Show Answer", key=f"show_answer_{current_index}")
            if show_answer:
                with st.container():
                    st.markdown("#### Back")
                    st.success(current_card['answer'])

                # FSRS rating interface
                quality = st.select_slider(
                    "How well did you remember this?",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: [
                        "Again (1)",
                        "Hard (2)",
                        "Good (3)",
                        "Easy (4)"
                    ][x-1],
                    key=f"quality_{current_index}"
                )

                if st.button("Submit Review", key=f"submit_{current_index}"):
                    handle_review_submission(current_card, quality, current_index, len(flashcards))

        with col_approfondimento:
            display_learning_features(st, model_choice, openai_client, current_card, current_index)

    except Exception as e:
        logger.error(f"Error in display_flashcard_content: {str(e)}")
        st.error("An error occurred while displaying the flashcard")
        if st.button("Reset Review Session"):
            st.session_state.current_deck = None
            st.session_state.current_flashcard_index = 0
            st.session_state.conversation_history = []
            st.session_state.show_answer = False
            st.rerun()

def handle_review_submission(card, quality, current_index, total_cards):
    """Handle the submission of a flashcard review"""
    try:
        review_time = datetime.now(timezone.utc)

        # Add to pending reviews
        if 'pending_reviews' not in st.session_state:
            st.session_state.pending_reviews = []

        st.session_state.pending_reviews.append({
            'card_id': card['id'],
            'quality': quality,
            'review_time': review_time
        })

        # Update session state
        st.session_state.reviewed_flashcards.append({
            'id': card['id'],
            'quality': quality,
            'topic_name': card.get('topic_name'),
            'review_time': review_time
        })

        logger.info(f"Added review for card {card['id']} with quality {quality}")

        # Check if this was the last card
        if current_index >= total_cards - 1:
            process_pending_reviews()
            st.session_state.session_complete = True
            st.success("🎉 Deck completed!")
        else:
            st.session_state.current_flashcard_index += 1

        st.rerun()

    except Exception as e:
        logger.error(f"Error in review submission: {str(e)}")
        st.error("An error occurred while submitting the review")

def process_pending_reviews():
    """Process all pending reviews in batch"""
    if not hasattr(st.session_state, 'pending_reviews'):
        return

    conn = get_conn()
    cur = conn.cursor()
    try:
        logger.info("Processing pending reviews...")
        for review in st.session_state.pending_reviews:
            # Get current card state
            cur.execute("""
                SELECT next_review FROM flashcards WHERE flashcard_id = %s
            """, (review['card_id'],))
            card_next_review = cur.fetchone()[0]

            # Update FSRS state
            fsrs_manager.update_flashcard_review(
                review['card_id'], 
                review['quality'], 
                review['review_time']
            )

            # Only mark as reviewed_today if next review is beyond today
            cur.execute("""
                UPDATE flashcards 
                SET reviewed_today = (next_review > (CURRENT_DATE + interval '1 day'))
                WHERE flashcard_id = %s
                RETURNING next_review, reviewed_today
            """, (review['card_id'],))

            new_next_review, is_reviewed = cur.fetchone()
            logger.info(f"Card {review['card_id']} new next_review: {new_next_review}, reviewed_today: {is_reviewed}")

        conn.commit()
        logger.info("Successfully processed all pending reviews")
        st.session_state.pending_reviews = []

    except Exception as e:
        conn.rollback()
        logger.error(f"Error processing pending reviews: {str(e)}")
        raise
    finally:
        cur.close()
        close_conn(conn)

def is_deck_completed(subject, topic, username):
    """Check if all cards in a deck have been reviewed today"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                COUNT(*) as total_cards,
                COUNT(CASE WHEN reviewed_today = true THEN 1 END) as reviewed_cards
            FROM flashcards f
            JOIN topics t ON f.topic_id = t.topic_id
            JOIN subjects s ON t.subject_id = s.subject_id
            WHERE f.username = %s
            AND s.name = %s
            AND t.name = %s
            AND DATE(next_review) = CURRENT_DATE
        """, (username, subject, topic))

        total, reviewed = cur.fetchone()
        return total > 0 and total == reviewed
    finally:
        cur.close()
        close_conn(conn)

def display_learning_features(st, model_choice, openai_client, current_card, current_index):
    """Display AI-powered learning features with state clearing"""
    st.markdown("### 💬 Learning Tools")

    # Clear previous responses when switching cards
    current_card_key = f"card_{current_card['id']}"
    if 'previous_card_key' not in st.session_state or st.session_state.previous_card_key != current_card_key:
        st.session_state.previous_card_key = current_card_key
        st.session_state.conversation_history = []
        for option in ["Explain", "Example", "Mnemonic"]:
            if option in st.session_state:
                st.session_state[option] = {"clicked": False, "response": None}

    context = get_rag_context(current_card['question'], current_card['answer'])

    # Predefined learning tools
    for option, emoji, func in [
        ("Explain", "🔍", explain_flashcard),
        ("Example", "📝", provide_example),
        ("Mnemonic", "🧠", create_mnemonic)
    ]:
        if option not in st.session_state:
            st.session_state[option] = {"clicked": False, "response": None}

        if st.button(f"{emoji} {option}", key=f"{option}_{current_card_key}"):
            handle_learning_tool(st, option, func, model_choice, openai_client, current_card, context)

        if st.session_state[option]["clicked"]:
            display_tool_response(st, option, current_index, model_choice)

    # Chat interface with state clearing
    display_chat_interface(st, model_choice, openai_client, current_card, current_index, context)

def display_tool_response(st, option, current_index, model_choice):
    """Display AI tool response with feedback options"""
    st.write(st.session_state[option]["response"]['content'])
    update_token_usage(st, st.session_state[option]["response"]['usage'], model_choice)

    with st.expander("📢 Feedback", expanded=False):
        display_ai_response_feedback(
            st, 
            f"{option.lower()}_{current_index}", 
            st.session_state[option]["response"]['content'], 
            model_choice
        )

def handle_learning_tool(st, option, func, model_choice, openai_client, card, context):
    """Handle the execution of a learning tool"""
    st.session_state[option]["clicked"] = True
    response = func(
        openai_client,
        model_choice,
        st.session_state.current_deck['subject'],
        st.session_state.difficulty,
        st.session_state.class_year,
        st.session_state.language,
        card,
        context
    )
    st.session_state[option]["response"] = response
    st.rerun()

def display_chat_interface(st, model_choice, openai_client, current_card, current_index, context):
    """Display the chat interface for the current flashcard"""
    st.markdown("#### 💭 Chat")

    for idx, (role, content) in enumerate(st.session_state.conversation_history):
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Tutor AI:** {content}")
            with st.expander("📢 Feedback", expanded=False):
                display_chat_feedback(st, idx, content, model_choice)

    user_input = st.text_input("Ask a question about this flashcard:", key="flashcard_follow_up")

    if st.button("Send", key="flashcard_send"):
        handle_chat_input(st, user_input, model_choice, openai_client, current_card, context)

    if st.button("🔄 Reset Chat"):
        st.session_state.conversation_history = []
        st.rerun()

def handle_chat_input(st, user_input, model_choice, openai_client, card, context):
    """Handle user chat input"""
    if not user_input:
        st.warning("Please enter a question.")
        return

    with st.spinner("Processing your question..."):
        try:
            response, usage = flash_chat(
                user_input,
                context,
                model_choice,
                openai_client,
                st.session_state.current_deck['subject'],
                st.session_state.class_year,
                st.session_state.language,
                st.session_state.difficulty,
                st.session_state.conversation_history,
                card
            )
            st.session_state.conversation_history.append(("user", user_input))
            st.session_state.conversation_history.append(("assistant", response))
            update_token_usage(st, usage, model_choice)
            st.rerun()
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

def display_review_statistics(st):
    """Display comprehensive review statistics"""
    st.title("📊 Flashcard Analytics")

    # Today's stats
    with st.expander("📅 Today's Review Statistics", expanded=True):
        today_stats = get_today_stats(st.session_state.username)

        if today_stats['reviewed'] > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cards Reviewed Today", today_stats['reviewed'])
            with col2:
                st.metric("Average Rating", f"{today_stats['avg_rating']:.2f}")
            with col3:
                st.metric("Completion Rate", f"{today_stats['completion_rate']:.1f}%")
        else:
            st.info("No cards reviewed today yet.")

    # Review heatmap
    with st.expander("🗓️ Review Activity Calendar", expanded=True):
        review_data = get_review_heatmap_data(st.session_state.username)
        if review_data:
            display_review_heatmap(review_data)
        else:
            st.info("No review history available yet.")

    # Future load projection
    with st.expander("📈 Review Load Projection", expanded=True):
        future_load = get_future_review_load(st.session_state.username)
        if future_load:
            display_review_projection(future_load)
        else:
            st.info("No future reviews scheduled yet.")

    # Performance trends
    with st.expander("📊 Performance Trends", expanded=True):
        display_performance_trends(st.session_state.username)

def get_today_stats(username):
    """Get today's review statistics"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            WITH today_stats AS (
                SELECT 
                    COUNT(DISTINCT fr.flashcard_id) as reviewed,
                    AVG(fr.rating) as avg_rating,
                    COUNT(DISTINCT fr.flashcard_id) * 100.0 / NULLIF(
                        (SELECT COUNT(*) FROM flashcards 
                         WHERE username = %s 
                         AND DATE(next_review) = CURRENT_DATE), 0
                    ) as completion_rate
                FROM flashcard_reviews fr
                WHERE fr.username = %s
                AND DATE(fr.review_time) = CURRENT_DATE
                AND fr.is_initial_review = FALSE
                AND fr.rating > 0
            )
            SELECT 
                COALESCE(reviewed, 0),
                COALESCE(avg_rating, 0),
                COALESCE(completion_rate, 0)
            FROM today_stats
        """, (username, username))
        result = cur.fetchone()
        return {
            'reviewed': result[0],
            'avg_rating': result[1],
            'completion_rate': result[2]
        }
    finally:
        cur.close()
        close_conn(conn)

def get_review_heatmap_data(username):
    """Get review activity data for heatmap"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                DATE(review_time) as review_date,
                COUNT(*) as review_count
            FROM flashcard_reviews
            WHERE username = %s
            AND review_time >= CURRENT_DATE - INTERVAL '365 days'
            AND NOT is_initial_review  -- Exclude initial creation reviews
            GROUP BY DATE(review_time)
            ORDER BY review_date
        """, (username,))

        results = cur.fetchall()
        logger.info(f"Retrieved {len(results)} days of review data for heatmap")
        return results
    except Exception as e:
        logger.error(f"Error getting review heatmap data: {str(e)}")
        return []
    finally:
        cur.close()
        close_conn(conn)

def display_review_heatmap(review_data):
    """Display review activity heatmap"""
    if not review_data:
        st.info("No review history available yet.")
        return

    try:
        # Convert to DataFrame and ensure proper datetime handling
        df = pd.DataFrame(review_data, columns=['date', 'count'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Get week and day numbers
        df['week'] = df.index.isocalendar().week
        df['day'] = df.index.dayofweek  # Use dayofweek instead of isocalendar().weekday

        logger.info(f"Created heatmap DataFrame with {len(df)} entries")

        # Create pivot table
        pivot_data = df.pivot_table(
            index='week',
            columns='day',
            values='count',
            aggfunc='sum',
            fill_value=0
        )

        # Create heatmap
        fig = px.imshow(
            pivot_data,
            labels={
                'y': 'Week of Year', 
                'x': 'Day of Week',
                'color': 'Reviews'
            },
            color_continuous_scale='Viridis',
            title='Review Activity Heatmap'
        )

        # Update layout
        fig.update_layout(
            xaxis_title='Day of Week',
            yaxis_title='Week of Year',
            xaxis=dict(
                ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                tickvals=[0, 1, 2, 3, 4, 5, 6]
            )
        )

        st.plotly_chart(fig)

    except Exception as e:
        logger.error(f"Error creating heatmap: {str(e)}")
        st.warning("Unable to display heatmap at this time.")

def get_future_review_load(username):
    """Get projected review load"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                DATE(next_review) as review_date,
                COUNT(*) as card_count,
                string_agg(DISTINCT t.name, ', ') as topics
            FROM flashcards f
            JOIN topics t ON f.topic_id = t.topic_id
            WHERE f.username = %s
            AND next_review > CURRENT_DATE
            GROUP BY DATE(next_review)
            ORDER BY review_date
            LIMIT 30
        """, (username,))
        return cur.fetchall()
    finally:
        cur.close()
        close_conn(conn)

def display_review_projection(future_load):
    """Display projected review load"""
    if not future_load:
        st.info("No future reviews scheduled.")
        return

    df = pd.DataFrame(future_load, columns=['date', 'count', 'topics'])

    # Line chart for card count
    fig = px.line(df, x='date', y='count', 
                  title='Projected Review Load',
                  labels={'count': 'Cards Due', 'date': 'Date'})
    st.plotly_chart(fig)

    # Detailed breakdown
    st.dataframe(df)

def display_performance_trends(username):
    """Display performance trends over time"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                DATE_TRUNC('day', review_time) as review_date,
                AVG(rating) as avg_rating,
                COUNT(*) as review_count,
                string_agg(DISTINCT t.name, ', ') as topics
            FROM flashcard_reviews fr
            JOIN flashcards f ON fr.flashcard_id = f.flashcard_id
            JOIN topics t ON f.topic_id = t.topic_id
            WHERE fr.username = %s
            GROUP BY DATE_TRUNC('day', review_time)
            ORDER BY review_date
        """, (username,))

        data = cur.fetchall()
        df = pd.DataFrame(data, columns=['date', 'rating', 'count', 'topics'])

        # Rating trend
        fig1 = px.line(df, x='date', y='rating',
                      title='Average Rating Trend',
                      labels={'rating': 'Average Rating', 'date': 'Date'})
        st.plotly_chart(fig1)

        # Review volume trend
        fig2 = px.bar(df, x='date', y='count',
                      title='Daily Review Volume',
                      labels={'count': 'Cards Reviewed', 'date': 'Date'})
        st.plotly_chart(fig2)

    finally:
        cur.close()
        close_conn(conn)

def get_topic_statistics(reviewed_cards):
    """Calculate statistics per topic"""
    topic_stats = {}
    for card in reviewed_cards:
        topic = card.get('topic_name', 'General')
        if topic not in topic_stats:
            topic_stats[topic] = {'count': 0, 'total_quality': 0}
        topic_stats[topic]['count'] += 1
        topic_stats[topic]['total_quality'] += card['quality']

    # Calculate averages
    for stats in topic_stats.values():
        stats['avg_quality'] = stats['total_quality'] / stats['count']

    return topic_stats

def display_review_schedule(username):
    """Display upcoming review schedule"""
    st.subheader("📅 Upcoming Reviews")
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                t.name as topic,
                COUNT(*) as card_count,
                MIN(next_review) as next_review,
                MAX(next_review) as last_review
            FROM flashcards f
            JOIN topics t ON f.topic_id = t.topic_id
            WHERE f.username = %s
            AND next_review > CURRENT_TIMESTAMP
            GROUP BY t.name
            ORDER BY MIN(next_review)
        """, (username,))

        schedule = cur.fetchall()
        if schedule:
            for topic, count, next_review, last_review in schedule:
                with st.expander(f"📘 {topic}"):
                    st.write(f"**Cards:** {count}")
                    st.write(f"**Next review:** {next_review.strftime('%d/%m/%Y')}")
                    st.write(f"**Review window:** {(last_review - next_review).days} days")
        else:
            st.info("No upcoming reviews scheduled.")
    finally:
        cur.close()
        close_conn(conn)

def display_today_review_status():
    """Display summary of today's review progress"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                COUNT(*) as total_due,
                COUNT(CASE WHEN EXISTS (
                    SELECT 1 
                    FROM flashcard_reviews fr 
                    WHERE fr.flashcard_id = f.flashcard_id 
                    AND DATE(fr.review_time) = CURRENT_DATE
                    AND fr.is_initial_review = FALSE
                ) THEN 1 END) as reviewed
            FROM flashcards f
            WHERE f.username = %s
            AND DATE(f.next_review) = CURRENT_DATE
        """, (st.session_state.username,))

        total_due, reviewed = cur.fetchone()

        if total_due > 0:
            progress = reviewed / total_due
            st.progress(progress)
            st.write(f"Today's progress: {reviewed}/{total_due} cards reviewed")

            if reviewed == total_due and reviewed > 0:
                st.success("🎉 All cards for today have been reviewed!")

    except Exception as e:
        logger.error(f"Error getting review status: {str(e)}")
    finally:
        cur.close()
        close_conn(conn)

# its okay for now i dont have the embeddings for that particular document saved so it wont work (have to readjust other things as well)
def get_rag_context(question, answer):
    """Get relevant context from RAG pipeline"""
    try:
        relevant_docs = retrieve_relevant_docs(
            st.session_state.rag_pipeline,
            f"{question} {answer}",
            k=5
        )
        return " ".join(doc.page_content for doc in relevant_docs)
    except Exception as e:
        logger.error(f"Error getting RAG context: {str(e)}")
        return ""

def update_token_usage(st, usage, model_choice):
    """Update token usage tracking"""
    st.session_state.token_tracker.update(usage, model_choice)
    cost = st.session_state.token_tracker.calculate_cost(usage, model_choice)
    log_usage(st.session_state.username, model_choice, usage, cost)

def get_available_decks(username):
    """Get flashcard decks that are due for review"""
    now = datetime.now(timezone.utc)
    conn = get_conn()
    cur = conn.cursor()
    try:
        logger.info(f"Fetching available decks at {now}")
        cur.execute("""
            WITH deck_counts AS (
                SELECT 
                    s.name as subject,
                    t.name as topic,
                    COUNT(*) as total_cards,
                    COUNT(CASE WHEN f.reviewed_today AND f.next_review > CURRENT_TIMESTAMP THEN 1 END) as reviewed_cards,
                    MIN(f.next_review) as next_review,
                    t.topic_id,
                    COUNT(CASE 
                        WHEN f.next_review <= CURRENT_TIMESTAMP 
                        THEN 1 
                    END) as due_now,
                    STRING_AGG(
                        CASE 
                            WHEN f.next_review <= CURRENT_TIMESTAMP + interval '24 hours'
                            AND f.next_review > CURRENT_TIMESTAMP
                            THEN f.next_review::text 
                        END,
                        ',' ORDER BY f.next_review
                    ) as upcoming_times
                FROM flashcards f
                JOIN topics t ON f.topic_id = t.topic_id
                JOIN subjects s ON t.subject_id = s.subject_id
                WHERE f.username = %s
                AND DATE(f.next_review) = CURRENT_DATE
                GROUP BY s.name, t.name, t.topic_id
            )
            SELECT 
                subject,
                topic,
                total_cards,
                reviewed_cards,
                next_review,
                topic_id,
                due_now,
                upcoming_times
            FROM deck_counts
            WHERE due_now > 0  -- Only show decks with cards due now
            ORDER BY next_review ASC
        """, (username,))

        deck_data = cur.fetchall()
        logger.info(f"Retrieved {len(deck_data)} decks for user {username}")
        for deck in deck_data:
            logger.info(f"Deck: {deck[0]}-{deck[1]}, Due now: {deck[6]}, Next review: {deck[4]}")
        return deck_data
    finally:
        cur.close()
        close_conn(conn)

def display_no_flashcards_message(all_flashcards):
    """Display message when no flashcards are available"""
    conn = get_conn()
    cur = conn.cursor()
    try:
        # Check for upcoming reviews
        cur.execute("""
            SELECT 
                MIN(next_review) as next_review,
                COUNT(*) as upcoming_count
            FROM flashcards
            WHERE username = %s
            AND DATE(next_review) > CURRENT_DATE
        """, (st.session_state.username,))
        next_review_data = cur.fetchone()

        if next_review_data and next_review_data[0]:
            next_review_date = next_review_data[0].date()
            upcoming_count = next_review_data[1]
            st.info(f"""
            ✅ All flashcards for today have been reviewed!

            Next review session: {next_review_date.strftime('%d/%m/%Y')} ({upcoming_count} cards)
            """)
        else:
            st.info("""
            No flashcards scheduled for review today.

            You can generate new flashcards by:
            1. Going to the **Materia** page
            2. Selecting a subject and topic
            3. Processing your study materials
            4. Generating flashcards
            """)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Go to Materia Page"):
                st.session_state.page = "materia"
                st.rerun()
        with col2:
            if st.button("View Study Calendar"):
                st.session_state.page = "attivita"
                st.rerun()

    finally:
        cur.close()
        close_conn(conn)

    # with col_approfondimento:
    #     st.markdown("### 💬 Approfondimento")

    #     # Get context from RAG pipeline
    #     context = get_rag_context(st.session_state.rag_pipeline, current_card['question']) #should also feed the answer as context not only the question
        
    #     # Predefined options as buttons
    #     for option, emoji, func in [
    #         ("Spiega", "🔍", explain_flashcard),
    #         ("Esempio", "📝", provide_example),
    #         ("Mnemotecnica", "🧠", create_mnemonic)
    #     ]:
    #         if option not in st.session_state:
    #             st.session_state[option] = {"clicked": False, "response": None}

    #         if st.button(f"{emoji} {option}"):
    #             st.session_state[option]["clicked"] = True
    #             response = func(openai_client, model_choice, st.session_state.subject, st.session_state.difficulty, st.session_state.class_year, st.session_state.language, current_card, context)
    #             st.session_state[option]["response"] = response
    #             st.rerun()

    #         if st.session_state[option]["clicked"]:
    #             st.write(st.session_state[option]["response"]['content'])
    #             update_token_usage(st, st.session_state[option]["response"]['usage'], model_choice)

    #             # Display feedback for this response
    #             with st.expander("📢 Lascia un feedback su questa risposta", expanded=False):
    #                 display_ai_response_feedback(st, f"{option.lower()}_{current_index}", st.session_state[option]["response"]['content'], model_choice)
        
    #     # Chat di approfondimento
    #     st.markdown("#### Chat di Approfondimento")
    #     for idx, (role, content) in enumerate(st.session_state.conversation_history):
    #         if role == "user":
    #             st.markdown(f"**Tu:** {content}")
    #         else:
    #             st.markdown(f"**Tutor AI:** {content}")
    #             with st.expander("📢 Feedback sulla risposta", expanded=False):
    #                 display_chat_feedback(st, idx, content, model_choice)
    #     user_input = st.text_input("Fai domande di approfondimento su questa flashcard:", key="flashcard_follow_up")
    #     if st.button("Invia", key="flashcard_send"):
    #         if user_input:
    #             with st.spinner("Il tutor sta elaborando la tua domanda..."):
    #                 try:
    #                     response, usage = flash_chat(
    #                         user_input,
    #                         st.session_state.rag_pipeline,
    #                         model_choice,
    #                         openai_client,
    #                         st.session_state.subject,
    #                         st.session_state.class_year,
    #                         st.session_state.language,
    #                         st.session_state.difficulty,
    #                         st.session_state.conversation_history,
    #                         current_card
    #                     )
    #                     st.session_state.conversation_history.append(("user", user_input))
    #                     st.session_state.conversation_history.append(("assistant", response))
    #                     update_token_usage(st, usage, model_choice)
    #                     st.rerun()
    #                 except Exception as e:
    #                     st.error(f"Errore durante la generazione della risposta: {str(e)}")
    #         else:
    #             st.warning("Per favore, inserisci una domanda.")
    #     if st.button("🔄 Resetta Chat Flashcard"):
    #         st.session_state.conversation_history = []
    #         st.rerun()
    # # Display feedback form if all flashcards are completed
    # #if st.session_state.get('all_flashcards_completed', False):
    #     #st.markdown("---")
    #     #st.subheader("📊 Feedback sulla Sessione di Flashcard")
    #     #display_flashcard_feedback(st, model_choice)
        
def display_ai_response_feedback(st, response_id, response, model):
    if f'ai_response_feedback_{response_id}' not in st.session_state:
        st.session_state[f'ai_response_feedback_{response_id}'] = {'rating': None, 'comment': ''}
    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("👍", key=f"thumbs_up_{response_id}"):
            st.session_state[f'ai_response_feedback_{response_id}']['rating'] = 1
    with cols[1]:
        if st.button("👎", key=f"thumbs_down_{response_id}"):
            st.session_state[f'ai_response_feedback_{response_id}']['rating'] = 0
    st.session_state[f'ai_response_feedback_{response_id}']['comment'] = st.text_input("Lascia un commento (opzionale)", key=f"comment_{response_id}")
    if st.button("Invia Feedback", key=f"submit_feedback_{response_id}"):
        if st.session_state[f'ai_response_feedback_{response_id}']['rating'] is not None:
            log_feedback(
                st.session_state.username,
                'ai_response',
                response_id,
                '',
                response,
                st.session_state[f'ai_response_feedback_{response_id}']['rating'],
                st.session_state[f'ai_response_feedback_{response_id}']['comment'],
                model,
                json.dumps(st.session_state.conversation_history)
            )
            st.success("Grazie per il tuo feedback!")
            st.session_state[f'ai_response_feedback_{response_id}'] = {'rating': None, 'comment': ''}
        else:
            st.warning("Per favore, seleziona un rating (👍 o 👎) prima di inviare il feedback.")

def display_chat_feedback(st, idx, response, model):
    if f'flashcard_chat_feedback_{idx}' not in st.session_state:
        st.session_state[f'flashcard_chat_feedback_{idx}'] = {'rating': None, 'comment': ''}

    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("👍", key=f"thumbs_up_flashcard_{idx}"):
            st.session_state[f'flashcard_chat_feedback_{idx}']['rating'] = 1
    with cols[1]:
        if st.button("👎", key=f"thumbs_down_flashcard_{idx}"):
            st.session_state[f'flashcard_chat_feedback_{idx}']['rating'] = 0

    st.session_state[f'flashcard_chat_feedback_{idx}']['comment'] = st.text_input("Lascia un commento (opzionale)", key=f"comment_flashcard_{idx}")

    if st.button("Invia Feedback", key=f"submit_flashcard_feedback_{idx}"):
        if st.session_state[f'flashcard_chat_feedback_{idx}']['rating'] is not None:
            log_feedback(
                st.session_state.username,
                'flashcard_chat',
                str(idx),
                st.session_state.conversation_history[idx*2]['content'] if idx > 0 else '',
                response,
                st.session_state[f'flashcard_chat_feedback_{idx}']['rating'],
                st.session_state[f'flashcard_chat_feedback_{idx}']['comment'],
                model,
                json.dumps(st.session_state.conversation_history)
            )
            st.success("Grazie per il tuo feedback!")
            st.session_state[f'flashcard_chat_feedback_{idx}'] = {'rating': None, 'comment': ''}
        else:
            st.warning("Per favore, seleziona un rating (👍 o 👎) prima di inviare il feedback.")

def display_flashcard_feedback(st, model_choice):
    with st.expander("📢 Feedback sulla sessione di flashcard", expanded=True):
        st.write("Ci piacerebbe sapere cosa pensi di questa sessione di flashcard!")

        if 'flashcard_session_feedback' not in st.session_state:
            st.session_state.flashcard_session_feedback = {'rating': 3, 'comment': ''}

        rating = st.slider("Come valuteresti questa sessione?", 1, 5, st.session_state.flashcard_session_feedback['rating'])
        st.session_state.flashcard_session_feedback['rating'] = rating

        comments = st.text_area("Hai suggerimenti per migliorare?", value=st.session_state.flashcard_session_feedback['comment'])
        st.session_state.flashcard_session_feedback['comment'] = comments

        if st.button("Invia Feedback Sessione"):
            log_feedback(
                st.session_state.username,
                'flashcard_session',
                '',
                'Flashcard Session',
                json.dumps(st.session_state.flashcards),
                rating,
                comments,
                model_choice,
                json.dumps(st.session_state.conversation_history)
            )
            st.success("Grazie per il tuo prezioso feedback!")
            # Reset the completion state and feedback inputs
            st.session_state.all_flashcards_completed = False
            st.session_state.flashcard_session_feedback = {'rating': 3, 'comment': ''}
            st.rerun()