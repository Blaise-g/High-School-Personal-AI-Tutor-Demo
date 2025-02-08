# admin_dashboard_page.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.usage_db import get_total_usage, get_usage_by_user
from utils.feedback_db import get_feedback_summary
from utils.auth import get_all_users

def admin_dashboard():
    st.title("📊 Dashboard Amministratore")
    if not st.session_state.get('is_admin', False):
        st.error("Non hai il permesso di accedere a questa pagina.")
        return

    # Overall Usage Statistics
    total_usage = get_total_usage()
    total_users = len(get_all_users())

    st.header("📈 Metriche Chiave di Performance (KPI)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Utenti Totali", total_users)
    col2.metric("Token Totali", f"{total_usage[0]:,}")
    col3.metric("Token Input", f"{total_usage[1]:,}")
    col4.metric("Token Output", f"{total_usage[2]:,}")
    col5.metric("Costo Totale", f"${total_usage[3]:.2f}")

    # Usage Over Time
    usage_data = get_usage_by_user()
    if usage_data:
        df = pd.DataFrame(usage_data, columns=[
            'username', 'timestamp', 'model', 'total_tokens', 'input_tokens', 'output_tokens', 'cost'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        st.header("📊 Analisi Temporale")
        # Daily usage trends
        daily_usage = df.groupby(df['timestamp'].dt.date).agg({
            'total_tokens': 'sum',
            'cost': 'sum',
            'username': 'nunique'
        }).reset_index()
        daily_usage.columns = ['date', 'total_tokens', 'cost', 'active_users']

        fig_trends = go.Figure()
        fig_trends.add_trace(go.Scatter(x=daily_usage['date'], y=daily_usage['total_tokens'], name='Token Totali', yaxis='y1'))
        fig_trends.add_trace(go.Scatter(x=daily_usage['date'], y=daily_usage['active_users'], name='Utenti Attivi', yaxis='y2'))
        fig_trends.add_trace(go.Scatter(x=daily_usage['date'], y=daily_usage['cost'], name='Costo', yaxis='y3'))
        fig_trends.update_layout(
            title='Tendenze di Utilizzo Giornaliero',
            yaxis=dict(title='Token Totali'),
            yaxis2=dict(title='Utenti Attivi', overlaying='y', side='right'),
            yaxis3=dict(title='Costo ($)', overlaying='y', side='right', anchor='free', position=1),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_trends, use_container_width=True)
        st.caption("Questo grafico mostra le tendenze giornaliere di token utilizzati, utenti attivi e costi.")

        st.header("👥 Engagement degli Utenti")
        # User engagement
        user_engagement = df.groupby('username').agg({
            'timestamp': 'count',
            'total_tokens': 'sum',
            'cost': 'sum'
        }).reset_index()
        user_engagement.columns = ['username', 'requests', 'total_tokens', 'cost']

        fig_engagement = px.scatter(user_engagement, x='requests', y='total_tokens', size='cost',
                                    hover_name='username', log_x=True, log_y=True,
                                    title="Engagement degli Utenti: Richieste vs Token Utilizzati")
        st.plotly_chart(fig_engagement, use_container_width=True)
        st.caption("Questo grafico a dispersione mostra la relazione tra il numero di richieste e i token utilizzati per ciascun utente. La dimensione dei punti rappresenta il costo associato.")

        # Update the Model Performance section
        st.header("🤖 Performance dei Modelli")
        model_performance = df.groupby('model').agg({
            'total_tokens': 'sum',
            'cost': 'sum',
            'username': 'nunique'
        }).reset_index()
        model_performance['cost_per_1M_tokens'] = model_performance['cost'] / model_performance['total_tokens'] * 1_000_000
        fig_model_perf = px.bar(model_performance, x='model', y=['total_tokens', 'cost'],
                                title="Performance dei Modelli: Utilizzo e Costo",
                                labels={'value': 'Valore', 'variable': 'Metrica'})
        st.plotly_chart(fig_model_perf, use_container_width=True)
        st.caption("Questo grafico a barre confronta l'utilizzo dei token e i costi associati per ciascun modello AI.")

        # Add a table showing cost per 1M tokens for each model
        st.subheader("Costo per 1M Token per Modello")
        st.dataframe(model_performance[['model', 'cost_per_1M_tokens']].sort_values('cost_per_1M_tokens', ascending=False))

        st.header("🔄 Ritenzione degli Utenti")
        # User retention (simplified)
        df['week'] = df['timestamp'].dt.isocalendar().week
        user_retention = df.groupby(['username', 'week']).size().unstack().notna().sum().reset_index()
        user_retention.columns = ['week', 'retained_users']

        fig_retention = px.line(user_retention, x='week', y='retained_users',
                                title="Ritenzione degli Utenti per Settimana")
        st.plotly_chart(fig_retention, use_container_width=True)
        st.caption("Questo grafico mostra il numero di utenti che ritornano ad utilizzare il servizio settimana dopo settimana.")

        st.header("🏆 Top 10 Utenti per Utilizzo")
        # Top users table
        top_users = user_engagement.sort_values('cost', ascending=False).head(10)
        st.dataframe(top_users)
        st.caption("Questa tabella mostra i 10 utenti con il maggior utilizzo del servizio in termini di costo.")

    # Feedback Summary
    st.header("📢 Riassunto Feedback")
    feedback_summary = get_feedback_summary()
    if feedback_summary:
        feedback_df = pd.DataFrame(feedback_summary, columns=['feature', 'total_feedback', 'positive_feedback', 'negative_feedback'])
        feedback_df['positive_rate'] = feedback_df['positive_feedback'] / feedback_df['total_feedback']
        st.dataframe(feedback_df)
        fig_feedback = px.bar(feedback_df, x='feature', y=['positive_feedback', 'negative_feedback'], 
                              title="Feedback per Funzionalità", barmode='group')
        st.plotly_chart(fig_feedback, use_container_width=True)
        st.caption("Questo grafico mostra la distribuzione di feedback positivi e negativi per ciascuna funzionalità del sistema.")
    else:
        st.write("Nessun feedback disponibile al momento.")

    # Export Data
    st.header("📤 Esporta Dati")
    if st.button("Esporta Dati di Utilizzo (CSV)"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="usage_data.csv",
            mime="text/csv"
        )
        st.caption("Clicca qui per scaricare un file CSV contenente tutti i dati di utilizzo.")