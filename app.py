import pandas as pd
import streamlit as st
from prophet import Prophet
import plotly.graph_objects as go
from datetime import timedelta
import numpy as np
from sklearn.metrics import mean_absolute_error

# Function to format large numbers
def format_number(num):
    if pd.isna(num) or num is None:
        return "N/A"
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    if num >= 1e6:
        return f"{num / 1e6:.2f}M"
    if num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return f"{num:.2f}"

# Function to calculate evaluation metrics
def calculate_metrics(actual, predicted):
    """Calculate MAE and R² metrics"""
    mae = mean_absolute_error(actual, predicted)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MAE': mae,
        'R²': r2
    }

# Load and clean the uploaded CSV data
@st.cache_data
def load_data(file, date_col, price_col, volume_col=None):
    try:
        df = pd.read_csv(file)
        # Rename columns to standard names
        df = df.rename(columns={date_col: 'date', price_col: 'price'})
        if volume_col and volume_col in df.columns:
            df = df.rename(columns={volume_col: 'volume'})
        # Convert date column to datetime and ensure no timezone
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.tz_localize(None)
        df = df.dropna(subset=['date', 'price'])
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        else:
            df['volume'] = 0  # Default volume to 0 if not provided
        df = df.dropna(subset=['price'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Streamlit app
st.title("Cryptocurrency Price Analysis and Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with cryptocurrency data", type=["csv"])
if uploaded_file is not None:
    # Load the CSV to inspect columns
    df_temp = pd.read_csv(uploaded_file)
    columns = df_temp.columns.tolist()
    
    # Column selection
    st.subheader("Select Column Names")
    date_col = st.selectbox("Select the Date column", options=columns, index=0)
    price_col = st.selectbox("Select the Price column", options=columns, index=1)
    volume_col = st.selectbox("Select the Volume column (optional)", options=[None] + columns, index=0)
    
    # Load and process data
    uploaded_file.seek(0)  # Reset file pointer
    df = load_data(uploaded_file, date_col, price_col, volume_col)
    
    if df is not None and not df.empty:
        # Display latest price
        latest_date = df['date'].max()
        latest_price = df[df['date'] == latest_date]['price'].iloc[0]
        st.subheader(f"Latest Price ({latest_date.strftime('%B %d, %Y')})")
        st.write(f"${format_number(latest_price)}")
        
        # Interesting fact (generic)
        st.subheader("Interesting Fact")
        st.write(
            "Cryptocurrency prices often exhibit weekly patterns: prices may dip on weekends due to lower trading activity, "
            "while weekdays, particularly midweek, can see higher prices driven by institutional trading and market news."
        )
        
        # Prepare data for Prophet
        prophet_df = df[['date', 'price', 'volume']].rename(columns={'date': 'ds', 'price': 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], errors='coerce', utc=True).dt.tz_localize(None)
        
        # Train Prophet model
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=15.0,
            seasonality_mode='multiplicative'
        )
        if 'volume' in prophet_df.columns and prophet_df['volume'].notna().any():
            model.add_regressor('volume')
        try:
            model.fit(prophet_df)
        except Exception as e:
            st.error(f"Error training Prophet model: {str(e)}")
            st.stop()
        
        # Create future dataframe for 6 months (180 days)
        future = model.make_future_dataframe(periods=180)
        future['volume'] = prophet_df['volume'].mean() if 'volume' in prophet_df.columns else 0
        forecast = model.predict(future)
        
        # Adjust predictions to align with latest price
        # Convert latest_date to same format as forecast ds column
        latest_date_normalized = pd.to_datetime(latest_date).normalize()
        
        # Find the forecast row that matches the latest date
        forecast_match = forecast[forecast['ds'].dt.normalize() == latest_date_normalized]
        if not forecast_match.empty:
            latest_forecast = forecast_match['yhat'].iloc[0]
            scaling_factor = latest_price / latest_forecast if latest_forecast != 0 else 1
            forecast['yhat'] *= scaling_factor
            forecast['yhat_lower'] *= scaling_factor
            forecast['yhat_upper'] *= scaling_factor
        
        # Calculate Model Performance Metrics
        st.subheader("Model Performance Metrics")
        
        # Get historical predictions (in-sample)
        historical_forecast = forecast[forecast['ds'] <= df['date'].max()]
        historical_data = prophet_df.merge(historical_forecast[['ds', 'yhat']], on='ds', how='inner')
        
        if not historical_data.empty:
            metrics = calculate_metrics(historical_data['y'], historical_data['yhat'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE (Mean Absolute Error)", f"{metrics['MAE']:,.2f}")
            with col2:
                st.metric("R² (Coefficient of Determination)", f"{metrics['R²']:.4f}")
                
                # Model quality interpretation
                if metrics['R²'] > 0.9:
                    quality = "Excellent"
                elif metrics['R²'] > 0.8:
                    quality = "Good"
                elif metrics['R²'] > 0.6:
                    quality = "Moderate"
                else:
                    quality = "Poor"
                st.metric("Model Quality", quality)
            
            # Additional metrics explanation
            with st.expander("📊 Metrics Explanation"):
                st.write("""
                
                **MAE (Mean Absolute Error)**: Average absolute difference between actual and predicted values. More robust to outliers.
              
                **R² (Coefficient of Determination)**: Proportion of variance explained by the model. Values closer to 1 indicate better fit.
                """)
        
        # Historical and Predicted Price Chart
        st.subheader("Historical and Predicted Price (6-Month Forecast)")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='Historical Price', line=dict(color="#06f111")
        ))
        fig1.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predicted Price', line=dict(color="#ff2a00")
        ))
        fig1.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', 
            line=dict(color="#e5e80e", dash='dash'), fill=None
        ))
        fig1.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', 
            line=dict(color="#cfe40d", dash='dash'), fill='tonexty', opacity=0.2
        ))
        fig1.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            yaxis_tickformat="$,.2f",
            template="plotly_white",
            height=400,
            font=dict(family="Arial", size=12),
            hovermode="x unified"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Residuals Analysis
        st.subheader("Model Residuals Analysis")
        if not historical_data.empty:
            historical_data['residuals'] = historical_data['y'] - historical_data['yhat']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Residuals over time
                fig_res1 = go.Figure()
                fig_res1.add_trace(go.Scatter(
                    x=historical_data['ds'], y=historical_data['residuals'], 
                    mode='lines+markers', name='Residuals', 
                    line=dict(color='#ff6b6b'), marker=dict(size=4)
                ))
                fig_res1.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                fig_res1.update_layout(
                    title="Residuals Over Time",
                    xaxis_title="Date",
                    yaxis_title="Residuals (Actual - Predicted)",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig_res1, use_container_width=True)
            
            with col2:
                # Residuals distribution
                fig_res2 = go.Figure()
                fig_res2.add_trace(go.Histogram(
                    x=historical_data['residuals'], 
                    name='Residuals Distribution',
                    marker_color='#4ecdc4',
                    opacity=0.7
                ))
                fig_res2.update_layout(
                    title="Residuals Distribution",
                    xaxis_title="Residuals",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig_res2, use_container_width=True)
        
        # Monthly Trends with Forecast
        st.subheader("Monthly Price Trends (Historical and Forecasted)")
        df['year_month'] = df['date'].dt.to_period('M').dt.to_timestamp()
        monthly_trends = df.groupby('year_month')['price'].mean().reset_index()
        
        # Create forecast monthly trends
        forecast['year_month'] = forecast['ds'].dt.to_period('M').dt.to_timestamp()
        future_data = forecast[forecast['ds'] > df['date'].max()]
        if not future_data.empty:
            forecast_monthly = future_data.groupby('year_month')['yhat'].mean().reset_index()
            monthly_trends = pd.concat([
                monthly_trends.rename(columns={'price': 'value', 'year_month': 'date'}),
                forecast_monthly.rename(columns={'yhat': 'value', 'year_month': 'date'})
            ]).reset_index(drop=True)
        else:
            monthly_trends = monthly_trends.rename(columns={'price': 'value', 'year_month': 'date'})
        
        monthly_trends['moving_avg'] = monthly_trends['value'].rolling(window=3, min_periods=1).mean()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=monthly_trends['date'], y=monthly_trends['value'], mode='lines+markers', 
            name='Monthly Average', line=dict(color='#82ca9d'), marker=dict(size=8)
        ))
        fig2.add_trace(go.Scatter(
            x=monthly_trends['date'], y=monthly_trends['moving_avg'], mode='lines', 
            name='3-Month Moving Avg', line=dict(color='#ff7300', dash='dash')
        ))
        fig2.add_annotation(
            x=latest_date, y=latest_price, text=f"Actual: ${format_number(latest_price)}", 
            showarrow=True, arrowhead=2, ax=20, ay=-30, font=dict(color="red", size=12)
        )
        fig2.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Price (USD)",
            yaxis_tickformat="$,.2f",
            template="plotly_white",
            height=400,
            font=dict(family="Arial", size=12),
            hovermode="x unified",
            showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Daily Trends (by Day of Week) with Volatility
        st.subheader("Daily Price Trends (by Day of Week)")
        df['day_of_week'] = df['date'].dt.day_name()
        daily_trends = df.groupby('day_of_week')['price'].agg(['mean', 'std']).reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ).reset_index()
        daily_trends['color'] = daily_trends['mean'].apply(lambda x: f'rgba(136, 132, 216, {x/daily_trends["mean"].max()})')
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=daily_trends['day_of_week'], y=daily_trends['mean'], 
            name='Average Price', marker_color=daily_trends['color'],
            error_y=dict(type='data', array=daily_trends['std'], visible=True)
        ))
        fig3.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Average Price (USD)",
            yaxis_tickformat="$,.2f",
            template="plotly_white",
            height=400,
            font=dict(family="Arial", size=12),
            hovermode="x unified",
            showlegend=True
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Yearly Trends with Forecast
        st.subheader("Yearly Price Trends (Historical and Forecasted)")
        df['year'] = df['date'].dt.year
        yearly_trends = df.groupby('year')['price'].mean().reset_index()
        
        # Add forecast for future years
        future_years = forecast[forecast['ds'] > df['date'].max()]
        if not future_years.empty:
            forecast_yearly = future_years.groupby(future_years['ds'].dt.year)['yhat'].mean().reset_index()
            forecast_yearly.columns = ['year', 'value']
            yearly_trends = pd.concat([
                yearly_trends.rename(columns={'price': 'value'}),
                forecast_yearly
            ]).reset_index(drop=True)
        else:
            yearly_trends = yearly_trends.rename(columns={'price': 'value'})
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=yearly_trends['year'], y=yearly_trends['value'], mode='lines+markers', 
            name='Yearly Average', line=dict(color='#ff7300'), marker=dict(size=10)
        ))
        
        # Add confidence interval for forecast years if available
        if not future_years.empty:
            forecast_max_year = future_years['ds'].dt.year.max()
            year_data = future_years[future_years['ds'].dt.year == forecast_max_year]
            if not year_data.empty:
                forecast_upper = year_data['yhat_upper'].mean()
                forecast_lower = year_data['yhat_lower'].mean()
                fig4.add_trace(go.Scatter(
                    x=[forecast_max_year, forecast_max_year], y=[forecast_lower, forecast_upper], mode='lines', 
                    name=f'{forecast_max_year} Confidence Interval', line=dict(color='#ff7300', dash='dash')
                ))
        
        fig4.add_annotation(
            x=latest_date.year, y=latest_price, text=f"Actual {latest_date.year}: ${format_number(latest_price)}", 
            showarrow=True, arrowhead=2, ax=20, ay=-30, font=dict(color="red", size=12)
        )
        fig4.update_layout(
            xaxis_title="Year",
            yaxis_title="Average Price (USD)",
            yaxis_tickformat="$,.2f",
            template="plotly_white",
            height=400,
            font=dict(family="Arial", size=12),
            hovermode="x unified",
            showlegend=True
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        # Summary and Conclusion
        st.subheader("Summary and Conclusion")
        peak_price = df['price'].max()
        peak_date = df[df['price'] == peak_price]['date'].iloc[0].strftime('%B %Y')
        
        # Get forecast values safely
        forecast_current = forecast[forecast['ds'].dt.normalize() == latest_date_normalized]
        forecast_latest = forecast_current['yhat'].iloc[0] if not forecast_current.empty else latest_price
        
        # Get next month forecast
        next_month_date = latest_date + timedelta(days=30)
        next_month_normalized = pd.to_datetime(next_month_date).normalize()
        forecast_next = forecast[forecast['ds'].dt.normalize() == next_month_normalized]
        forecast_next_month = forecast_next['yhat'].iloc[0] if not forecast_next.empty else forecast_latest
        
        # Include model performance in summary
        model_performance_summary = ""
        if not historical_data.empty:
            model_performance_summary = (f" The Prophet model demonstrates {quality.lower()} performance with an R² of "
                                       f"{metrics['R²']:.3f}.")
        
        st.write(
            f"The cryptocurrency price data from {df['date'].min().strftime('%B %Y')} to {latest_date.strftime('%B %Y')} "
            f"shows significant volatility, with a peak of ${format_number(peak_price)} in {peak_date}. "
            f"The actual price on {latest_date.strftime('%B %d, %Y')} was ${format_number(latest_price)}. "
            f"The tuned Prophet model predicts a price of approximately ${format_number(forecast_latest)} for "
            f"{latest_date.strftime('%B %Y')} and ${format_number(forecast_next_month)} for "
            f"{(latest_date + timedelta(days=30)).strftime('%B %Y')}, with a 6-month forecast extending to "
            f"{(latest_date + timedelta(days=180)).strftime('%B %Y')}.{model_performance_summary} Enhanced monthly and yearly trend charts include "
            f"forecasted data, showing continued growth with seasonal patterns. Daily trends reveal slight variations, "
            f"with weekends showing lower prices and higher volatility."
        )
    else:
        st.error("Failed to load or process the dataset. Please ensure it contains valid date and price columns.")
else:
    st.info("Please upload a CSV file to begin analysis.")