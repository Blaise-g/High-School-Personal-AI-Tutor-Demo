import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.auth import is_logged_in
from utils.usage_db import get_user_usage
from utils.feedback_db import get_user_feedback

def user_usage_page():
    st.title("📊 Il Tuo Resoconto di Utilizzo")
    if not is_logged_in():
        st.error("Per favore, effettua il login per visualizzare il tuo resoconto di utilizzo.")
        return
    usage_data = get_user_usage(st.session_state.username)
    feedback_data = get_user_feedback(st.session_state.username)
    if usage_data:
        df = pd.DataFrame(usage_data, columns=[
            'username', 'timestamp', 'model',
            'total_tokens', 'input_tokens', 'output_tokens', 'cost'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        if df.empty:
            st.warning("Non ci sono dati di utilizzo validi disponibili.")
            return
        st.header("💡 Riepilogo Utilizzo")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Costo Totale", f"${df['cost'].sum():.4f}")
        col2.metric("Token Totali", f"{df['total_tokens'].sum():,}")
        col3.metric("Richieste Totali", f"{len(df):,}")
        col4.metric("Costo Medio per Richiesta", f"${df['cost'].mean():.4f}")
        st.header("📈 Analisi Temporale")

        # Usage over time (daily)
        daily_usage = df.groupby(df['timestamp'].dt.date).agg({
            'total_tokens': 'sum',
            'cost': 'sum',
            'username': 'count'
        }).reset_index()
        daily_usage.columns = ['date', 'total_tokens', 'cost', 'requests']
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(x=daily_usage['date'], y=daily_usage['total_tokens'], name='Token Totali', yaxis='y1'))
        fig_daily.add_trace(go.Scatter(x=daily_usage['date'], y=daily_usage['requests'], name='Richieste', yaxis='y2'))
        fig_daily.update_layout(
            title='Utilizzo Giornaliero: Token e Richieste',
            yaxis=dict(title='Token Totali'),
            yaxis2=dict(title='Richieste', overlaying='y', side='right'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_daily, use_container_width=True)
        st.caption("Questo grafico mostra l'andamento giornaliero dei token utilizzati e delle richieste effettuate.")
        # Cost over time (daily)
        fig_cost = px.line(daily_usage, x='date', y='cost', 
                           title="Costo Giornaliero",
                           labels={'cost': 'Costo ($)', 'date': 'Data'})
        fig_cost.update_layout(yaxis_title="Costo ($)")
        st.plotly_chart(fig_cost, use_container_width=True)
        st.caption("Questo grafico illustra l'andamento dei costi giornalieri associati all'utilizzo del servizio.")
        st.header("🔍 Analisi dei Modelli")
        # Usage distribution by model
        model_usage = df.groupby('model').agg({
            'total_tokens': 'sum', 
            'cost': 'sum', 
            'username': 'count'
        }).reset_index()
        model_usage.columns = ['model', 'total_tokens', 'cost', 'requests']
        fig_model = px.sunburst(model_usage, path=['model'], values='total_tokens',
                                title="Distribuzione dell'Utilizzo per Modello")
        st.plotly_chart(fig_model, use_container_width=True)
        st.caption("Questo grafico a torta mostra la distribuzione dell'utilizzo dei token tra i diversi modelli AI.")
        st.header("📊 Metriche di Efficienza")
        col1, col2 = st.columns(2)
        col1.metric("Token per Richiesta", f"{df['total_tokens'].mean():.2f}")
        # Update this line to use cost per 1M tokens
        col2.metric("Costo per 1M Token", f"${(df['cost'].sum() / df['total_tokens'].sum() * 1_000_000):.4f}")
        st.caption("Queste metriche mostrano l'efficienza media dell'utilizzo in termini di token per richiesta e costo per 1 milione di token.")
        st.header("📅 Modelli di Utilizzo")
        # Usage patterns
        df['hour'] = df['timestamp'].dt.hour
        hourly_usage = df.groupby('hour')['total_tokens'].mean().reset_index()
        fig_hourly = px.bar(hourly_usage, x='hour', y='total_tokens',
                            title="Utilizzo Medio dei Token per Ora del Giorno",
                            labels={'total_tokens': 'Token Medi', 'hour': 'Ora'})
        st.plotly_chart(fig_hourly, use_container_width=True)
        st.caption("Questo grafico mostra la distribuzione media dell'utilizzo dei token durante le diverse ore del giorno.")
        st.header("📄 Utilizzo Recente")
        st.dataframe(df.head(10).sort_values('timestamp', ascending=False))
        st.caption("Questa tabella mostra le tue 10 interazioni più recenti con il sistema.")
    else:
        st.write("Nessun dato di utilizzo disponibile al momento.")
    if feedback_data:
        st.header("🗣️ Il Tuo Feedback")
        feedback_df = pd.DataFrame(feedback_data, columns=['timestamp', 'feature', 'rating', 'comments'])
        feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
        # Display feedback summary
        st.subheader("Riassunto del tuo feedback:")
        feedback_summary = feedback_df.groupby('feature').agg({
            'rating': ['mean', 'count'],
            'comments': 'count'
        }).reset_index()
        feedback_summary.columns = ['Feature', 'Rating Medio', 'Totale Feedback', 'Commenti']
        st.dataframe(feedback_summary)
        # Visualize feedback distribution
        fig_feedback = px.bar(feedback_summary, x='Feature', y='Rating Medio', 
                              title="Distribuzione dei Rating per Feature",
                              color='Totale Feedback', 
                              labels={'Rating Medio': 'Rating Medio', 'Feature': 'Funzionalità'})
        st.plotly_chart(fig_feedback, use_container_width=True)
        st.caption("Questo grafico mostra la distribuzione dei tuoi rating medi per ciascuna funzionalità del sistema.")
        # Display recent feedback
        st.subheader("I tuoi feedback più recenti:")
        st.dataframe(feedback_df.sort_values('timestamp', ascending=False).head(5))
        st.caption("Questa tabella mostra i tuoi 5 feedback più recenti.")
    else:
        st.write("Nessun feedback fornito al momento.")