import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
from fpdf import FPDF
import io

# Ensure NLTK resources are available
@st.cache_resource
def download_nltk_data():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()

# Streamlit Page Config
st.set_page_config(
    page_title="📊 Brand Reputation Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.8rem;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
.info-box {
    background-color: #e8f4f8;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3498db;
    margin: 1rem 0;
}
.success-box {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
}
.metric-container {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize Sentiment Analyzer
sia = download_nltk_data()

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = False

# Sidebar Navigation with icons and descriptions
st.sidebar.markdown("## 🧭 Navigation")
st.sidebar.markdown("---")
menu_options = {
    "🏠 Home": "Upload and process your data",
    "📊 Analysis": "View brand reputation scores",
    "📈 Trends": "Explore sentiment over time",
    "📄 Report": "Download your analysis"
}

menu = st.sidebar.radio(
    "Choose a section:",
    list(menu_options.keys()),
    format_func=lambda x: x,
    help="Navigate through different sections of the analysis"
)

# Display description in sidebar
st.sidebar.markdown(f"**{menu_options[menu]}**")

# Add help section in sidebar
with st.sidebar.expander("❓ Need Help?"):
    st.markdown("""
    **Required CSV Format:**
    - Column 1: `brand` (brand name)
    - Column 2: `review` (customer review text)
    - Optional: `date` (for trend analysis)
    
    **Example:**
    ```
    brand,review,date
    Apple,Great product!,2024-01-15
    Samsung,Not satisfied,2024-01-16
    ```
    """)

# ------------------ FUNCTIONS ------------------
@st.cache_data
def load_data(file):
    """Load and validate CSV data"""
    try:
        df = pd.read_csv(file)
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def analyze_sentiment(text):
    """Analyze sentiment of text"""
    if pd.isna(text):
        return 0
    return sia.polarity_scores(str(text))["compound"]

@st.cache_data
def process_data(df):
    """Process the uploaded data with sentiment analysis"""
    try:
        # Validate required columns
        required_cols = ['brand', 'review']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return None, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Process sentiment
        df_processed = df.copy()
        
        with st.spinner("🔄 Analyzing sentiment... This may take a moment."):
            progress_bar = st.progress(0)
            total_rows = len(df_processed)
            
            sentiment_scores = []
            for i, text in enumerate(df_processed["review"]):
                sentiment_scores.append(analyze_sentiment(text))
                progress_bar.progress((i + 1) / total_rows)
            
            df_processed["Sentiment Score"] = sentiment_scores
            progress_bar.empty()
        
        return df_processed, None
    except Exception as e:
        return None, str(e)

def get_sentiment_label(score, include_emoji=True):
    """Convert sentiment score to readable label"""
    if include_emoji:
        if score >= 0.1:
            return "😊 Positive"
        elif score <= -0.1:
            return "😞 Negative"
        else:
            return "😐 Neutral"
    else:
        if score >= 0.1:
            return "Positive"
        elif score <= -0.1:
            return "Negative"
        else:
            return "Neutral"

def clean_text_for_pdf(text, max_length=None):
    """Clean text to remove characters that can't be encoded in latin-1"""
    if pd.isna(text):
        return ""
    
    # Convert to string
    text_str = str(text)
    
    # Replace common problematic characters
    replacements = {
        '\u2019': "'",  # Right single quotation mark
        '\u2018': "'",  # Left single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...',  # Horizontal ellipsis
        '\u00a0': ' ',  # Non-breaking space
        '\u2022': '-',  # Bullet point
    }
    
    for unicode_char, replacement in replacements.items():
        text_str = text_str.replace(unicode_char, replacement)
    
    # Remove any remaining non-latin-1 characters
    try:
        text_str.encode('latin-1')
        if max_length:
            return text_str[:max_length]
        return text_str
    except UnicodeEncodeError:
        # If still problematic, keep only ASCII characters
        clean_str = ''.join(char for char in text_str if ord(char) < 128)
        if max_length:
            return clean_str[:max_length]
        return clean_str

def generate_pdf(summary_df, trends_fig):
    """Generate PDF report"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Header
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 15, "Brand Reputation Analysis Report", ln=True, align="C")
        pdf.ln(5)
        
        # Subtitle with timestamp
        pdf.set_font("Arial", size=10)
        current_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        pdf.cell(0, 8, f"Generated on: {current_time}", ln=True, align="C")
        pdf.ln(10)
        
        # Executive Summary Section
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Executive Summary", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", size=11)
        total_brands = len(summary_df)
        avg_score = summary_df['Score'].mean()
        min_score = summary_df['Score'].min()
        max_score = summary_df['Score'].max()
        
        # Create comprehensive summary
        positive_brands = len(summary_df[summary_df['Score'] > 0.1])
        neutral_brands = len(summary_df[(summary_df['Score'] >= -0.1) & (summary_df['Score'] <= 0.1)])
        negative_brands = len(summary_df[summary_df['Score'] < -0.1])
        
        summary_lines = [
            f"This report analyzes {total_brands} brands with an average sentiment score of {avg_score:.3f}.",
            f"Sentiment scores range from {min_score:.3f} (lowest) to {max_score:.3f} (highest).",
            f"Brand Distribution: {positive_brands} positive, {neutral_brands} neutral, {negative_brands} negative.",
        ]
        
        for line in summary_lines:
            clean_line = clean_text_for_pdf(line)
            pdf.multi_cell(0, 6, clean_line)
            pdf.ln(2)
        
        pdf.ln(10)
        
        # Brand Performance Section
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Brand Performance Rankings", ln=True)
        pdf.ln(5)
        
        # Table header with better spacing
        pdf.set_font("Arial", "B", 10)
        col_widths = [50, 30, 35, 40]
        headers = ["Brand Name", "Score", "Sentiment", "Rating"]
        
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 8, header, 1, 0, "C")
        pdf.ln()
        
        # Table content
        pdf.set_font("Arial", size=9)
        for index, row in summary_df.iterrows():
            sentiment_label = get_sentiment_label(row['Score'], include_emoji=False)
            
            # Determine rating
            score = float(row['Score'])
            if score >= 0.5:
                rating = "Excellent"
            elif score >= 0.1:
                rating = "Good"
            elif score >= -0.1:
                rating = "Average"
            else:
                rating = "Poor"
            
            # Clean and limit brand name
            brand_name = clean_text_for_pdf(row['Brand'], max_length=20)
            if not brand_name.strip():  # If cleaning removed everything
                brand_name = f"Brand_{index + 1}"
            
            # Table row
            pdf.cell(col_widths[0], 7, brand_name, 1, 0, "L")
            pdf.cell(col_widths[1], 7, f"{score:.3f}", 1, 0, "C")
            pdf.cell(col_widths[2], 7, sentiment_label, 1, 0, "C")
            pdf.cell(col_widths[3], 7, rating, 1, 0, "C")
            pdf.ln()
        
        pdf.ln(15)
        
        # Scoring Guide Section
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Scoring Guide", ln=True)
        pdf.ln(3)
        
        pdf.set_font("Arial", size=9)
        scoring_guide = [
            "Sentiment Score Ranges:",
            "- Excellent: 0.50 to 1.00 (Highly positive sentiment)",
            "- Good: 0.10 to 0.49 (Positive sentiment)", 
            "- Average: -0.10 to 0.10 (Neutral sentiment)",
            "- Poor: -1.00 to -0.11 (Negative sentiment)"
        ]
        
        for line in scoring_guide:
            clean_line = clean_text_for_pdf(line)
            pdf.cell(0, 5, clean_line, ln=True)
        
        pdf.ln(10)
        
        # Methodology Section
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Methodology", ln=True)
        pdf.ln(3)
        
        pdf.set_font("Arial", size=9)
        methodology_lines = [
            "This analysis uses VADER (Valence Aware Dictionary and sEntiment Reasoner),",
            "a lexicon and rule-based sentiment analysis tool specifically attuned to",
            "social media text. VADER analyzes text and returns sentiment scores ranging",
            "from -1.0 (most negative) to +1.0 (most positive).",
            "",
            "The tool considers:",
            "- Lexical features (positive/negative words)",
            "- Grammatical and syntactical rules",
            "- Capitalization, punctuation, and emoticons",
            "- Context and intensity modifiers"
        ]
        
        for line in methodology_lines:
            if line:  # Skip empty lines for spacing
                clean_line = clean_text_for_pdf(line)
                pdf.cell(0, 5, clean_line, ln=True)
            else:
                pdf.ln(3)
        
        pdf.ln(15)
        
        # Footer
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 5, "Report generated by Brand Reputation Analysis Tool", ln=True, align="C")
        
        # Return PDF as bytes with proper error handling
        try:
            pdf_output = pdf.output(dest="S")
            if isinstance(pdf_output, str):
                return pdf_output.encode("latin-1", errors="replace")
            return pdf_output
        except Exception as output_error:
            st.error(f"Error in PDF output: {str(output_error)}")
            return None
        
    except Exception as e:
        st.error(f"Error in PDF generation: {str(e)}")
        # Provide more specific error information
        if "unicode" in str(e).lower() or "codec" in str(e).lower():
            st.info("💡 This error is caused by special characters in your data. The system is cleaning the text and trying again.")
        return None

# ------------------ HOME PAGE ------------------
if menu == "🏠 Home":
    st.markdown('<h1 class="main-header">📊 Brand Reputation Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🚀 Welcome to Brand Reputation Analysis!</h3>
    <p>This tool helps you analyze customer sentiment across different brands. Simply upload your CSV file with customer reviews and get instant insights into brand performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section
    st.markdown('<h2 class="section-header">📂 Upload Your Data</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose your CSV file",
            type=["csv"],
            help="Upload a CSV file with 'brand' and 'review' columns"
        )
        
        if uploaded_file:
            # Show file info
            st.info(f"📁 **File:** {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Load and validate data
            df, error = load_data(uploaded_file)
            
            if error:
                st.error(f"❌ Error loading file: {error}")
            else:
                st.success(f"✅ File loaded successfully! Found {len(df)} records.")
                
                # Show data preview
                with st.expander("👀 Preview your data", expanded=True):
                    st.dataframe(df.head(), use_container_width=True)
                
                # Process data
                processed_df, process_error = process_data(df)
                
                if process_error:
                    st.error(f"❌ Error processing data: {process_error}")
                else:
                    st.session_state.df = processed_df
                    st.session_state.processed_data = True
                    
                    st.markdown("""
                    <div class="success-box">
                    <h4>🎉 Data processed successfully!</h4>
                    <p>Your data has been analyzed for sentiment. You can now explore the results in the Analysis and Trends sections.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 CSV Format Requirements")
        st.markdown("""
        **Required columns:**
        - `brand`: Brand name
        - `review`: Customer review text
        
        **Optional columns:**
        - `date`: For trend analysis
        
        **Example CSV:**
        ```
        brand,review,date
        Apple,Amazing phone!,2024-01-15
        Samsung,Poor quality,2024-01-16
        Apple,Love it,2024-01-17
        ```
        """)
        
        # Sample data download
        sample_data = pd.DataFrame({
            'brand': ['Apple', 'Samsung', 'Apple', 'Samsung', 'Google'],
            'review': [
                'Excellent product, highly recommended!',
                'Not satisfied with the quality',
                'Great customer service',
                'Amazing features and design',
                'Poor user experience'
            ],
            'date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19']
        })
        
        csv_sample = sample_data.to_csv(index=False)
        st.download_button(
            "📥 Download Sample CSV",
            csv_sample,
            "sample_brand_data.csv",
            "text/csv",
            help="Download a sample CSV file to see the expected format"
        )

# ------------------ ANALYSIS PAGE ------------------
elif menu == "📊 Analysis":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload and process a file in the Home section first.")
        st.markdown("👈 Navigate to **Home** to get started!")
    else:
        st.markdown('<h1 class="main-header">📈 Brand Reputation Analysis</h1>', unsafe_allow_html=True)
        
        df = st.session_state.df
        
        # Calculate summary statistics
        summary = df.groupby("brand")["Sentiment Score"].agg(['mean', 'count', 'std']).reset_index()
        summary.columns = ["Brand", "Score", "Reviews", "Std_Dev"]
        summary = summary.sort_values("Score", ascending=False)
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", len(df))
        with col2:
            st.metric("Brands Analyzed", len(summary))
        with col3:
            avg_sentiment = df["Sentiment Score"].mean()
            st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
        with col4:
            positive_reviews = len(df[df["Sentiment Score"] > 0.1])
            st.metric("Positive Reviews", f"{positive_reviews}/{len(df)}")
        
        st.markdown("---")
        
        # Brand performance cards
        st.markdown('<h2 class="section-header">🏆 Brand Performance Overview</h2>', unsafe_allow_html=True)
        
        # Create columns for brand cards
        num_brands = len(summary)
        cols_per_row = min(4, num_brands)
        rows_needed = (num_brands + cols_per_row - 1) // cols_per_row
        
        for row in range(rows_needed):
            cols = st.columns(cols_per_row)
            start_idx = row * cols_per_row
            end_idx = min(start_idx + cols_per_row, num_brands)
            
            for i, (_, brand_data) in enumerate(summary.iloc[start_idx:end_idx].iterrows()):
                with cols[i]:
                    sentiment_label = get_sentiment_label(brand_data["Score"], include_emoji=True)
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <h3 style="margin: 0; color: white;">{brand_data["Brand"]}</h3>
                        <h2 style="margin: 0.5rem 0; color: white;">{brand_data["Score"]:.3f}</h2>
                        <p style="margin: 0; opacity: 0.9;">{sentiment_label}</p>
                        <small style="opacity: 0.8;">{brand_data["Reviews"]} reviews</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h2 class="section-header">📊 Detailed Analysis</h2>', unsafe_allow_html=True)
            
            # Interactive bar chart
            fig = px.bar(
                summary, 
                x="Brand", 
                y="Score", 
                color="Score",
                color_continuous_scale="RdYlGn",
                title="Brand Reputation Scores",
                hover_data={'Reviews': True, 'Std_Dev': ':.3f'}
            )
            fig.update_layout(
                xaxis_title="Brand",
                yaxis_title="Average Sentiment Score",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<h2 class="section-header">📋 Summary Table</h2>', unsafe_allow_html=True)
            
            # Format the summary table
            display_summary = summary.copy()
            display_summary["Score"] = display_summary["Score"].round(3)
            display_summary["Std_Dev"] = display_summary["Std_Dev"].round(3)
            display_summary["Sentiment"] = display_summary["Score"].apply(lambda x: get_sentiment_label(x, include_emoji=True))
            
            st.dataframe(
                display_summary[["Brand", "Score", "Sentiment", "Reviews"]],
                use_container_width=True,
                hide_index=True
            )
        
        # Review distribution
        st.markdown('<h2 class="section-header">📈 Review Distribution</h2>', unsafe_allow_html=True)
        
        # Sentiment distribution pie chart
        sentiment_dist = df["Sentiment Score"].apply(
            lambda x: "Positive" if x > 0.1 else "Negative" if x < -0.1 else "Neutral"
        ).value_counts()
        
        fig_pie = px.pie(
            values=sentiment_dist.values,
            names=sentiment_dist.index,
            title="Overall Sentiment Distribution",
            color_discrete_map={
                "Positive": "#28a745",
                "Neutral": "#ffc107", 
                "Negative": "#dc3545"
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ------------------ TRENDS PAGE ------------------
elif menu == "📈 Trends":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload and process a file in the Home section first.")
        st.markdown("👈 Navigate to **Home** to get started!")
    else:
        st.markdown('<h1 class="main-header">📅 Sentiment Trends Over Time</h1>', unsafe_allow_html=True)
        
        df = st.session_state.df.copy()
        
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
                
                if len(df) == 0:
                    st.error("❌ No valid dates found in the dataset.")
                else:
                    # Date range info
                    date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
                    st.info(f"📅 **Analysis Period:** {date_range}")
                    
                    # Time aggregation options
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        time_period = st.selectbox(
                            "Time Aggregation",
                            ["Daily", "Weekly", "Monthly"],
                            help="Choose how to group the data over time"
                        )
                    
                    # Aggregate data based on selection
                    if time_period == "Weekly":
                        df["period"] = df["date"].dt.to_period("W").dt.start_time
                    elif time_period == "Monthly":
                        df["period"] = df["date"].dt.to_period("M").dt.start_time
                    else:
                        df["period"] = df["date"]
                    
                    trends = df.groupby(["period", "brand"])["Sentiment Score"].mean().reset_index()
                    
                    if len(trends) > 0:
                        # Main trend chart
                        fig = px.line(
                            trends, 
                            x="period", 
                            y="Sentiment Score", 
                            color="brand",
                            title=f"Sentiment Trends Over Time ({time_period})",
                            markers=True
                        )
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Average Sentiment Score",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Additional insights
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<h3 class="section-header">📊 Trend Summary</h3>', unsafe_allow_html=True)
                            
                            for brand in trends["brand"].unique():
                                brand_trends = trends[trends["brand"] == brand].sort_values("period")
                                if len(brand_trends) >= 2:
                                    start_score = brand_trends.iloc[0]["Sentiment Score"]
                                    end_score = brand_trends.iloc[-1]["Sentiment Score"]
                                    change = end_score - start_score
                                    
                                    trend_emoji = "📈" if change > 0.05 else "📉" if change < -0.05 else "➡️"
                                    
                                    st.write(f"{trend_emoji} **{brand}**: {change:+.3f} change")
                        
                        with col2:
                            # Volatility analysis
                            st.markdown('<h3 class="section-header">📉 Volatility Analysis</h3>', unsafe_allow_html=True)
                            
                            volatility = trends.groupby("brand")["Sentiment Score"].std().sort_values(ascending=False)
                            
                            for brand, vol in volatility.head().items():
                                stability = "🔴 High" if vol > 0.2 else "🟡 Medium" if vol > 0.1 else "🟢 Low"
                                st.write(f"**{brand}**: {stability} volatility ({vol:.3f})")
                    
                    else:
                        st.warning("⚠️ No data available for the selected time period.")
            
            except Exception as e:
                st.error(f"❌ Error processing date column: {str(e)}")
                st.info("💡 Make sure your date column is in a standard format (YYYY-MM-DD, MM/DD/YYYY, etc.)")
        else:
            st.markdown("""
            <div class="info-box">
            <h3>📅 No Date Column Found</h3>
            <p>To view sentiment trends over time, your CSV file needs to include a 'date' column.</p>
            <p><strong>Example format:</strong></p>
            <pre>brand,review,date
Apple,Great product!,2024-01-15
Samsung,Not satisfied,2024-01-16</pre>
            </div>
            """, unsafe_allow_html=True)

# ------------------ DOWNLOAD REPORT PAGE ------------------
elif menu == "📄 Report":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload and process a file in the Home section first.")
        st.markdown("👈 Navigate to **Home** to get started!")
    else:
        st.markdown('<h1 class="main-header">📄 Download Analysis Report</h1>', unsafe_allow_html=True)
        
        df = st.session_state.df
        summary = df.groupby("brand")["Sentiment Score"].mean().reset_index()
        summary.columns = ["Brand", "Score"]
        summary = summary.sort_values("Score", ascending=False)
        
        # Report preview
        st.markdown('<h2 class="section-header">📋 Report Preview</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Your report will include:**")
            st.markdown("""
            - 📊 Brand performance summary
            - 📈 Sentiment scores for each brand  
            - 📋 Analysis methodology notes
            - 📅 Generation timestamp
            """)
            
            st.markdown("**Summary of Results:**")
            for _, row in summary.iterrows():
                sentiment_label = get_sentiment_label(row['Score'], include_emoji=True)
                st.write(f"• **{row['Brand']}**: {row['Score']:.3f} {sentiment_label}")
        
        with col2:
            st.markdown("**Report Details:**")
            st.info(f"""
            📊 **Brands Analyzed:** {len(summary)}  
            📝 **Total Reviews:** {len(df)}  
            📅 **Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            """)
        
        # Generate and download
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🔄 Generate Report", type="primary", use_container_width=True):
                with st.spinner("📝 Generating your report..."):
                    try:
                        pdf_bytes = generate_pdf(summary, None)
                        
                        if pdf_bytes:
                            st.success("✅ Report generated successfully!")
                            
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_bytes,
                                file_name=f"Brand_Reputation_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf",
                                type="primary",
                                use_container_width=True
                            )
                        else:
                            st.error("❌ Failed to generate PDF report. Please try again.")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)}")
                        st.info("💡 Try downloading the CSV exports instead, or contact support if the issue persists.")
        
        # Additional export options
        st.markdown("---")
        st.markdown('<h3 class="section-header">📤 Additional Export Options</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export processed data
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📊 Download Processed Data (CSV)",
                csv_data,
                f"processed_sentiment_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                help="Download the original data with sentiment scores added"
            )
        
        with col2:
            # Export summary
            summary_csv = summary.to_csv(index=False)
            st.download_button(
                "📈 Download Summary (CSV)",
                summary_csv,
                f"brand_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                help="Download just the brand performance summary"
            )