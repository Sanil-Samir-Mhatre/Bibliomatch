import streamlit as st
import pandas as pd
import numpy as np
import os
from recommender import recommend_from_books, load_artifacts

st.set_page_config(page_title='BiblioMatch', layout='wide')

# Local data path
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'books.csv')

def load_data():
    return pd.read_csv(DATA_PATH)

def page_eda():
    st.title('📊 Preprocessing & Exploratory Data Analysis')
    st.markdown('### Understanding the BiblioMatch Dataset')
    
    st.write("Before building the recommendation models, we performed data cleaning and exploratory data analysis to understand the distributions and patterns in our book dataset.")
    
    try:
        df = load_data()
        
        st.subheader("1. Dataset Overview")
        st.write(f"**Total Books:** {len(df)}")
        st.dataframe(df.head(5))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Categories")
            # Filter out missing categories and count
            cat_counts = df['categories'].dropna().value_counts().head(10)
            st.bar_chart(cat_counts)
            
        with col2:
            st.subheader("Publication Year Distribution")
            # Filter valid years
            # Safely convert to numeric, coercing errors to NaN
            years = pd.to_numeric(df['published_year'], errors='coerce').dropna()
            years = years[(years > 1900) & (years <= 2026)]
            year_counts = years.value_counts().sort_index()
            st.line_chart(year_counts)
            
        st.subheader("Average Rating Distribution")
        # Creating a simple histogram using bar_chart
        ratings = pd.to_numeric(df['average_rating'], errors='coerce').dropna()
        # Binning ratings
        bins = pd.cut(ratings, bins=np.arange(0, 5.5, 0.5))
        rating_counts = bins.value_counts().sort_index()
        # Convert index to string for plotting
        rating_counts.index = rating_counts.index.astype(str)
        st.bar_chart(rating_counts)
        
        st.subheader("Preprocessing Steps")
        st.markdown("""
        - **Handling Missing Values:** Dropped rows with missing titles or essential metadata.
        - **Text Cleaning:** Normalized descriptions, titles, and authors by converting to lowercase and removing special characters.
        - **TF-IDF Vectorization:** Converted book descriptions and metadata into numerical vectors.
        - **Word Embeddings:** Trained Word2Vec on the corpus to capture semantic similarity.
        """)
        
    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_PATH}. Please ensure the data directory exists.")
    except Exception as e:
        st.error(f"An error occurred while analyzing the data: {e}")

def page_evaluation():
    st.title('📈 Evaluation Metrics')
    st.markdown('### Model Performance Comparison')
    
    st.write("We evaluated multiple approaches for content-based recommendation. Below are the offline evaluation metrics based on Precision@10 and Recall@10.")
    
    # Data from eval/results.md
    eval_data = {
        'Model': ['TF-IDF + Rerank (Primary)', 'Word2Vec (Ablation)', 'Roziq et al. 2026 (Ref)', 'Nuipian et al. 2023 (Ref)'],
        'Precision@10': [0.5895, 0.2368, 0.8000, 0.7000],
        'Recall@10': [0.0578, 0.0032, 0.5714, 0.7300]
    }
    
    eval_df = pd.DataFrame(eval_data)
    
    st.dataframe(eval_df, use_container_width=True)
    
    # Prepare data for plotting
    plot_data = eval_df.set_index('Model')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Precision@10 Comparison")
        st.bar_chart(plot_data['Precision@10'])
        
    with col2:
        st.subheader("Recall@10 Comparison")
        st.bar_chart(plot_data['Recall@10'])
        
    st.info("**TF-IDF + Rerank** is our primary local model. The reference models (Roziq et al. and Nuipian et al.) are shown as state-of-the-art benchmarks.")
    
    st.markdown("---")
    st.subheader("Simulated Confusion Matrices")
    st.write("Based on the Precision and Recall metrics, here is an approximation of the confusion matrices.")
    
    # Synthetic CMs derived from P & R
    cms = {
        'TF-IDF + Rerank': [[6, 4], [94, 896]],
        'Word2Vec': [[1, 3], [99, 897]],
        'Roziq et al. 2026': [[57, 14], [43, 886]],
        'Nuipian et al. 2023': [[73, 31], [27, 869]]
    }
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    col_a, col_b = st.columns(2)
    for idx, (model_name, matrix) in enumerate(cms.items()):
        df_cm = pd.DataFrame(matrix, index=['Relevant', 'Not Relevant'], columns=['Relevant', 'Not Relevant'])
        target_col = col_a if idx % 2 == 0 else col_b
        
        with target_col:
            st.markdown(f"**{model_name}**")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, 
                        annot_kws={"size": 12})
            ax.set_ylabel('True Label', fontweight='bold')
            ax.set_xlabel('Predicted Label', fontweight='bold')
            ax.set_title(f"{model_name}", fontsize=10)
            st.pyplot(fig)
            plt.close(fig)

def page_application():
    st.title('📚 BiblioMatch: Content Recommender')
    st.markdown('### Find your next favorite book.')

    try:
        # Load model artifacts
        with st.spinner('Loading recommendation engine...'):
            vectorizer, matrix, df = load_artifacts()

        selected_books = st.multiselect('Search for books you have read:', options=df['title'].unique())

        if st.button('Get Recommendations'):
            if selected_books:
                results = recommend_from_books(selected_books)
                if not results:
                    st.warning("Could not find enough data to recommend books based on your selection.")
                else:
                    for res in results:
                        with st.container():
                            st.write(f"**{res['title']}** by {res['authors']}")
                            st.caption(f"Category: {res['categories']} | Rating: {res['average_rating']}")
                            st.info(f"Similarity Score: {res['sim_score']:.2f}")
                            st.divider()
            else:
                st.warning('Please select at least one book.')
    except Exception as e:
        st.error(f"Error loading the recommendation artifacts: {e}")
        st.info("Please make sure you have the correct files in the 'artifacts' directory.")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["1. Preprocessing & EDA", "2. Evaluation Metrics", "3. Final Application"])

if page == "1. Preprocessing & EDA":
    page_eda()
elif page == "2. Evaluation Metrics":
    page_evaluation()
else:
    page_application()
