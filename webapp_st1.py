import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Helper Functions to Load Resources
# ---------------------------
@st.cache_resource
def load_model():
    """Load the pre-trained model from a pickle file."""
    with open("rf_opt_model2.pickle", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    """Load the dataset for insights and visualizations."""
    try:
        df = pd.read_csv('V4. Final Feature Engineering.csv')
        return df
    except Exception as e:
        st.error("Dataset not found. Please ensure 'V4. Final Feature Engineering.csv' is in the working directory.")
        return None


# ---------------------------
# Page 1: Exploratory Data Analaysis (EDA)
# ---------------------------
def eda():
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("---")
    st.subheader("For any data science project, starting with comprehensive Exploratory Data Analysis is a must. We need to examine and understand the dataset thoroughly before proceeding.")
    st.markdown("---")
    model = load_model()
    df = load_data()

    # List of numerical feature column names
    numerical = ["Duration (seconds)", "Publish Date", "Publish Time (24h)", "Views", "Likes", 
                    "Comments", "Channel Subscribers", "Total Videos Posted", "Channel Views", 
                    "Channel Creation Date", "Title Length", "Hashtag Count", 
                    "Video Age (days)", "Engagement Rate", "Channel Title Length", "Average Video Views",
                    "Channel Age (days)", "Sentiment Polarity"]
    df_numerical = df[numerical]
    
    # List of categorical feature column names
    categorical = ["Video ID", "Title", "Description", "Category", "Default Audio Language",
                   "Channel Title", "Channel Country", "Description Present", "Duration Category", 
                   "Day of Week", "Month", "Quarter", "Year", "Channel Size", "Category Definition", 
                   "Audio Language Name", "Channel Country Name", "Publish Hour", "Time of Day",
                   "Sentiment Category"]
    if df is not None:
        df_numerical = df[numerical]
    else:
        st.error("Data failed to load. Please check the source.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Descriptive Statistics", "Feature Elements","Correlation Heatmap", "Feature Distributions", "Scatter Plot"])
    with tab1:
        # Descriptive Statistics for Numerical Features in Transformed Dataset
        st.markdown(
            """
            Explore key statistics of numerical data at a glance. This section breaks down your dataset with insights into averages (**mean**), spread (**standard deviation**), and distribution percentiles (**min, 25%, 50%, 75%, max**). Uncover trends, outliers, and data patterns effortlessly.
            """
        )
        pd.set_option('display.float_format', '{:.2f}'.format)
        # Assuming 'transformed_df' is your processed dataset
        # Replace transformed_df with the actual dataset variable
        st.dataframe(df_numerical.describe().drop("count"), use_container_width=True)
    with tab2:
        # Feature Selection & Value Counts with Encoded Dictionary
        #st.subheader("Feature Distribution")
        st.markdown(
            """
            Discover the makeup of categorical data at a glance. This section highlights the count of each unique category, revealing distribution patterns, dominant values, and key insights into the dataset’s composition.
            """
        )
        feature_names = df.columns.tolist()
        Chosen_Features = st.selectbox(
            "Choose a feature to see its element count", 
            feature_names,
            help="Select a feature to display its unique elements and their respective counts."
        )
        
        # Retrieve value counts
        value_counts = df[Chosen_Features].value_counts().reset_index()
        value_counts.columns = ['Element', 'Count']
        value_counts.index = np.arange(1, len(value_counts) + 1)  # Set index to start from 1
        st.dataframe(value_counts, use_container_width=True, hide_index=True)
    with tab3:
        # Define the features for correlation analysis
        correlation_features = [
            "Duration (seconds)", "Views", "Likes", "Comments", "Channel Subscribers", 
            "Total Videos Posted", "Channel Views", "Title Length", "Hashtag Count", 
            "Video Age (days)", "Engagement Rate", "Channel Title Length", "Average Video Views", 
            "Channel Age (days)", "Sentiment Polarity"
        ]
        
        # Create a DataFrame for correlation analysis
        df_corr = df[correlation_features]
        
        st.markdown(
            """
            Explore the interrelationships between features using the correlation heatmap. 
            This visualization highlights the strength and direction of associations among video metrics, 
            helping to identify which attributes most influence views and engagement.
            """
        )
        
        # Create the heatmap figure
        #fig2, ax2 = plt.subplots(figsize=(10, 8))
        #corr_matrix = df_corr.corr()
        #sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
        #st.pyplot(fig2)
        st.image("correlation_heatmap.png")

    with tab4:
        st.markdown(
            """
            Analyze how each feature is distributed across the dataset. Select a feature to view its histogram and gain insights into its impact on video performance.
            """
        )
        features = df_corr.columns.tolist()
        selected_feature = st.selectbox("Select a feature to view its distribution", features, index=0)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_feature], kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_feature}")
        st.pyplot(fig)
    with tab5:
        # List of numerical feature column names
        scatter = ["Duration (seconds)", "Views", "Likes", "Comments", "Channel Subscribers", 
                   "Total Videos Posted", "Channel Views", "Title Length", "Hashtag Count", 
                   "Video Age (days)", "Engagement Rate", "Channel Title Length", "Average Video Views", 
                   "Channel Age (days)", "Sentiment Polarity",
                   "Description Present", "Duration Category", "Day of Week", 
                   "Month", "Quarter", "Year", "Channel Size", "Category Definition", "Audio Language Name", 
                   "Channel Country Name", "Publish Hour", "Time of Day", "Sentiment Category"]
        df_scatter = df[scatter]
        
        st.markdown(
            """
            Compare two features side by side using a scatter plot. Select features for the X and Y axes to uncover potential trends and patterns in video performance.
            """
        )
        features_scatter = df_scatter.columns.tolist()
        col1, col2 = st.columns(2)
        x_feature = col1.selectbox("Select feature for X-axis", features_scatter, index=0)
        y_feature = col2.selectbox("Select feature for Y-axis", features_scatter, index=1)

        fig3, ax3 = plt.subplots()
        sns.scatterplot(x=df[x_feature], y=df[y_feature], ax=ax3)
        ax3.set_title(f"Scatter Plot: {x_feature} vs {y_feature}")
        st.pyplot(fig3)

# ---------------------------
# Page 2: Features Selection
# ---------------------------
def features_selection():
    st.title("Features Selection")
    st.markdown("---")
    # Section 1: 3-Column Table of Eliminated Features
    st.header("1. Irrelevant Features Elimination")
    st.markdown(
        """
        Non-essential and redundant features are identified and removed to enhance model efficiency, reduce noise, and improve predictive performance.
        """
    )
    
    # Feature lists with renamed columns
    column1 = pd.DataFrame({'Irrelevant': ['Video ID', 'Title', 'Description', 'Category']})
    column2 = pd.DataFrame({'Irrelevant': ['Publish Date', 'Publish Time (24h)', 'Default Audio Language', 'Channel Title']})
    column3 = pd.DataFrame({'Irrelevant': ['Channel Creation Date', 'Channel Country', 'Year', '']})  # Added an empty string for alignment
    
    # Create three columns in Streamlit
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.dataframe(column1, use_container_width=True, hide_index=True)
    with col2:
        st.dataframe(column2, use_container_width=True, hide_index=True)
    with col3:
        st.dataframe(column3, use_container_width=True, hide_index=True)

    
    # Section 2: Feature Importances with gradient highlighting on 'Sum' and 'Average'
    data_importance = {
        "Feature": [
            "Likes", "Engagement_Rate", "Comments", "Average_Video_Views", "Hashtag_Count", 
            "Channel_Country", "Video_Age", "Weekly_Videos", "Total_Videos_Posted", "Title_Length", 
            "Month", "Channel_Subscribers", "Publish_Hour_Time", "Title_Sentiment", "Duration", 
            "Channel_Title_Length", "Quarter", "Channel_Age", "Description_Present", "Category", 
            "Audio_Language", "Channel_Views", "Weekday", "Monthly_Videos", "Channel_Size", 
            "Sentiment_Category", "Duration_Category"
        ],
        "Decision Tree": [
            0.745143, 0.240011, 0.008945, 0.002767, 0.000061, 0.000005, 0.000046, 0.000000, 
            0.000208, 0.000602, 0.000741, 0.000162, 0.000003, 0.000342, 0.000221, 0.000005, 
            0.000292, 0.000270, 0.000000, 0.000003, 0.000059, 0.000008, 0.000074, 0.000018, 
            0.000001, 0.000012, 0.000000
        ],
        "Random Forest": [
            0.737107, 0.226129, 0.011668, 0.002923, 0.000705, 0.000981, 0.001612, 0.000795, 
            0.001239, 0.001742, 0.001045, 0.001206, 0.001262, 0.000856, 0.001616, 0.001396, 
            0.000201, 0.001627, 0.000134, 0.001272, 0.001092, 0.001199, 0.000620, 0.000636, 
            0.000419, 0.000332, 0.000188
        ],
        "XGBoost": [
            0.658759, 0.154942, 0.022301, 0.010412, 0.013795, 0.013344, 0.010963, 0.011485, 
            0.010470, 0.007789, 0.008340, 0.008519, 0.007950, 0.007872, 0.006648, 0.007012, 
            0.007632, 0.006165, 0.007222, 0.004878, 0.004783, 0.004222, 0.004496, 0.000000, 
            0.000000, 0.000000, 0.000000
        ]
    }
    
    # Compute 'Sum' and 'Average' for each feature
    dt = data_importance["Decision Tree"]
    rf = data_importance["Random Forest"]
    xgb = data_importance["XGBoost"]
    sum_values = [round(a + b + c, 6) for a, b, c in zip(dt, rf, xgb)]
    avg_values = [round(s / 3, 6) for s in sum_values]
    data_importance["Sum"] = sum_values
    data_importance["Average"] = avg_values
    
    importance_df = pd.DataFrame(data_importance).set_index("Feature")
    
    st.markdown("---")
    st.subheader("Realizing that 27 features are too many, we decided to reduce them to focus on the most impactful ones.")
    st.markdown("---")
    st.header("2. Key Features Selection")
    st.subheader("A. Features Importance")
    st.markdown(
        """
        Key features are evaluated based on their importance across multiple models to refine the selection process, enhance model efficiency, and ensure a more efficient and user-friendly experience, especially for form-based interactions.
        """
    )
    styled_importance_df = importance_df.style.background_gradient(subset=["Sum", "Average"], cmap="viridis")
    st.dataframe(styled_importance_df, use_container_width=True)

    # Section 3: 3-Column Table of Selected Feature
    st.subheader("B. Result")
    st.markdown(
        """
        Based on feature importance across multiple models, the top 10 features have been selected for the final dataset to streamline analysis and improve model performance.
        """
    )

    
    # Feature lists with renamed columns
    column4 = pd.DataFrame({'Selected': ['Likes', 'Engagement_Rate', 'Comments', 'Average_Video_Views', 'Hashtag_Count']})
    column5 = pd.DataFrame({'Selected': ['Channel_Country', 'Video_Age', 'Weekly_Videos', 'Total_Videos_Posted', 'Title_Length']})
    
    # Create three columns in Streamlit
    col4, col5 = st.columns(2)
    with col4:
        st.dataframe(column4, use_container_width=True, hide_index=True)
    with col5:
        st.dataframe(column5, use_container_width=True, hide_index=True)

# ---------------------------
# Page 3: Model Development
# ---------------------------
def model_dev():
    st.title("Model Development & Evaluation")
    st.markdown("---")
    st.subheader("Now that our data is well-prepared, it's time to develop our machine learning model.")
    st.markdown("---")
    st.subheader("1. Model Training")
    with st.expander("Why We Chose Decision Tree, Random Forest & XGBoost Regressor?"):
        why_model = pd.DataFrame({
                "Main Points": ["Effective Handling of Categorical Variables", "Capturing Non-linear Relationships", 
                                "Robustness to Outliers and Noise", "Built-in Feature Importance", "Ensemble Benefits",
                                "Minimal Preprocessing Requirements", "Flexibility in Hyperparameter Tuning"],
                "Descriptions": ["These models naturally handle categorical features better than traditional linear regression or distance-based models like Support Vector Regression, as they split data based on feature thresholds without requiring extensive encoding or scaling.", 
                                 "Decision trees and their ensemble variants can model complex, non-linear interactions between features, providing flexibility that linear models lack.", 
                                 "Tree-based models are less sensitive to outliers because they partition the data into homogenous groups, making them robust when the data contains anomalies or noise.",
                                 "They offer insights into which features are most influential through feature importance metrics, aiding in model interpretability and further feature selection.",
                                 "Random Forest and XGBoost combine multiple trees to reduce variance and bias, respectively, leading to improved predictive performance over a single decision tree.",
                                 "They generally require less data preprocessing (e.g., scaling or normalization) compared to other algorithms, streamlining the modeling process.",
                                 "These models provide a wide range of hyperparameters that can be tuned to optimize performance for specific datasets and tasks."]
            })
        st.table(why_model)
    with st.expander("Our Approach"):
        st.image("model_development.png")
        approach = pd.DataFrame({
                "Main Points": ["Comprehensive Evaluation: Train-Validate-Test Approach",
                                "Maximizing Data Utility with Cross-Validation (K-Fold)", 
                                "Balancing Overfitting and Underfitting with Hyperparameter Tuning"],
                "Descriptions": ["Our approach uses a three-way split—train, validate, and test—to ensure robust model selection. Cross-validation is applied to tune hyperparameters, while the validation set allows us to compare different models and select a champion, with the test set ultimately confirming the model’s generalization capability.",
                                 "Given that our dataset consists of just over 6000 samples—a relatively small amount—we employed k-fold cross-validation to maximize its potential. By averaging out the randomness inherent in any single train-test split, cross-validation provides a more accurate measure of model performance and ensures robust evaluation.", 
                                 "To achieve the right balance between overfitting and underfitting, we fine-tuned model hyperparameters using grid search. This systematic approach helps identify the optimal settings that allow our models to fit the training data well while generalizing effectively to new data."]
            })
        st.table(approach)
        
    with st.expander("Dataset Split"):
        # Create a DataFrame with dataset split information
        st.caption(
            """
            This table summarizes the dataset split, showing the number of samples and percentage allocation for training, validation, and testing sets.
            """
        )
        data = {
            "Model": ["Train", "Validate", "Test"],
            "Number of Samples": [4161, 1387, 1388],
            "Percentage": ["60%", "20%", "20%"]
        }
        df_samples = pd.DataFrame(data)
        
        # Display the DataFrame in Streamlit
        st.dataframe(df_samples, use_container_width=True, hide_index=True)

    # Streamlit UI
    st.markdown("---")
    st.subheader("2. Model Evaluation & Selection")
    st.markdown("Now, let's assess model performance and choose the top performer")
    
    # Model performance data
    comparison = {"Model": ["Decision Tree", "Random Forest", "XGBoost"],
                  "Dataset": ["Validate", "Validate", "Validate"],
                  "Mean Absolute Error": [842559.637, 497808.709, 539798.812],
                  "Root Mean Squared Error": [2724559.324, 1356981.906, 1813183.213],
                  "Mean Absolute Percentage Error": [0.270, 0.209, 0.671],
                  "R2 Score": [0.944, 0.973, 0.975]}
    # Convert to DataFrame
    df_validate = pd.DataFrame(comparison)
    
    # Display the dataframe
    st.dataframe(df_validate, use_container_width=True, hide_index=True)
    st.subheader("Result: Based on the validation scores, Random Forest delivers the best performance.")

    st.markdown("---")
    st.subheader("3. Model Future Performance")
    st.markdown("With the champion model chosen, we now evaluate its performance on unseen data (test data).")
    # Model performance data
    test = {"Model": "Random Forest",
            "Dataset": "Test",
            "Mean Absolute Error": 595201.194,
            "Root Mean Squared Error": 3134901.134,
            "Mean Absolute Percentage Error": 0.183,
            "R2 Score": 0.942}
    # Convert to DataFrame
    df_test = pd.DataFrame([test])
    st.dataframe(df_test, use_container_width=True, hide_index=True)
    st.subheader("Result: Evaluation metrics indicate that the model performs exceptionally well on test data.")
# ---------------------------
# Page 4: Predictive Modelling 
# ---------------------------
import streamlit as st
import pandas as pd
import numpy as np

def predictive_modelling():
    st.title("YouTube Views Prediction")
    st.markdown("---")
    st.subheader(
        """
        By analyzing selected video attributes, our regression model estimates how many views your YouTube video may receive. Enter your video's details and click Submit to get a tailored prediction
        """
    )
    st.markdown("---")
    st.subheader("Enter Video Details:")
    
    # Mapping for categorical features: Channel Country
    channel_country_options = {
        'Algeria': 0, 'Argentina': 1, 'Australia': 2, 'Austria': 3, 'Azerbaijan': 4, 'Bangladesh': 5, 
        'Belarus': 6, 'Belgium': 7, 'Brazil': 8, 'Bulgaria': 9, 'Cambodia': 10, 'Canada': 11, 'Chile': 12, 
        'China': 13, 'Colombia': 14, 'Costa Rica': 15, 'Croatia': 16, 'Cyprus': 17, 'Czech Republic': 18, 
        'Denmark': 19, 'Dominican Republic': 20, 'Ecuador': 21, 'Egypt': 22, 'Estonia': 23, 'Finland': 24, 
        'France': 25, 'Georgia': 26, 'Germany': 27, 'Greece': 28, 'Hong Kong': 29, 'Hungary': 30, 'Iceland': 31, 
        'India': 32, 'Indonesia': 33, 'Iraq': 34, 'Ireland': 35, 'Israel': 36, 'Italy': 37, 'Japan': 38, 
        'Jordan': 39, 'Kazakhstan': 40, 'Kenya': 41, 'Latvia': 42, 'Lithuania': 43, 'Luxembourg': 44, 
        'Malaysia': 45, 'Malta': 46, 'Mexico': 47, 'Montenegro': 48, 'Morocco': 49, 'Nepal': 50, 'Netherlands': 51, 
        'New Zealand': 52, 'Nicaragua': 53, 'Nigeria': 54, 'Norway': 55, 'Oman': 56, 'Pakistan': 57, 'Panama': 58, 
        'Peru': 59, 'Philippines': 60, 'Poland': 61, 'Portugal': 62, 'Qatar': 63, 'Romania': 64, 'Russia': 65, 
        'Saudi Arabia': 66, 'Senegal': 67, 'Serbia': 68, 'Singapore': 69, 'Slovakia': 70, 'Slovenia': 71, 
        'South Africa': 72, 'South Korea': 73, 'Spain': 74, 'Sri Lanka': 75, 'Sweden': 76, 'Switzerland': 77, 
        'Taiwan': 78, 'Tanzania': 79, 'Thailand': 80, 'Tunisia': 81, 'Turkey': 82, 'Ukraine': 83, 
        'United Arab Emirates': 84, 'United Kingdom': 85, 'United States': 86, 'Unknown': 87, 'Vietnam': 88
    }

    # Create a form for manual input
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            Likes = st.number_input(
                "Likes", min_value=0, value=150000,
                help="Number of likes the video has received."
            )
            Engagement_Rate = st.number_input(
                "Engagement Rate (%)", min_value=0.0, value=5.0,
                help="Percentage representing (Likes + Comments) divided by Views."
            )
            Comments = st.number_input(
                "Comments", min_value=0, value=2500,
                help="Total number of comments on the video."
            )
            Average_Video_Views = st.number_input(
                "Average Video Views", min_value=0.0, value=1500000.0,
                help="Average views per video, calculated as Channel Views divided by Total Videos Posted."
            )
            Hashtag_Count = st.number_input(
                "Hashtag Count", min_value=0, value=3,
                help="Number of hashtags included in the video title."
            )
            
        with col2:
            Channel_Country = st.selectbox(
                "Channel Country", list(channel_country_options.keys()),
                help="Country where the channel was created."
            )
            Video_Age = st.number_input(
                "Video Age (days)", min_value=0, value=500,
                help="Number of days since the video was published."
            )
            Total_Videos_Posted = st.number_input(
                "Total Videos Posted", min_value=0, value=6000,
                help="Total count of videos uploaded by the channel."
            )
            Weekly_Videos = st.number_input(
                "Weekly Videos", min_value=0.0, value=5.0,
                help="Average number of videos posted per week, derived from Total Videos Posted and Channel Age."
            )
            Title_Length = st.number_input(
                "Title Length", min_value=0, value=50,
                help="Number of characters in the video title."
            )

        submit_button = st.form_submit_button(label="Predict Views")

        if submit_button:
            # Create input dictionary
            input_data = {
                'Likes': Likes,
                'Engagement_Rate': Engagement_Rate,
                'Comments': Comments,
                'Average_Video_Views': Average_Video_Views,
                'Hashtag_Count': Hashtag_Count,
                'Channel_Country': channel_country_options[Channel_Country],
                'Video_Age': Video_Age,
                'Weekly_Videos': Weekly_Videos,
                'Total_Videos_Posted': Total_Videos_Posted,
                'Title_Length': Title_Length
            }
            
            input_df = pd.DataFrame([input_data])

            # Load the pre-trained model (Assuming function exists)
            model = load_model()  
            prediction = model.predict(input_df)
            
            st.success("Prediction Complete!")
            colA, colB = st.columns(2)
            colA.markdown("#### Predicted Number of Views:")
            colB.markdown(f"#### {np.round(prediction[0], 0)}")
    
# ---------------------------
# Sidebar Navigation
# ---------------------------
page = st.sidebar.selectbox("Navigation", 
    ["Exploratory Data Analysis", "Features Selection", "Model Development & Evaluation", "Predictive Modelling"]
)

if page == "Exploratory Data Analysis":
    eda()
elif page == "Features Selection":
    features_selection()
elif page == "Model Development & Evaluation":
    model_dev()
elif page == "Predictive Modelling":
    predictive_modelling()

# Sidebar Table of Contents
st.sidebar.title("Table of Contents")
st.sidebar.markdown("""
- **1. Exploratory Data Analysis (EDA)**
  - Descriptive Statistics
  - Feature Elements
  - Correlation Heatmap
  - Features Distribution
  - Scatter Plot
- **2. Features Selection**
  - Irrelevant Features Elimination
  - Key Features Selection
- **3. Model Development & Evaluation**
  - Model Training
  - Model Evaluation & Selection
  - Model Future Performance
- **4. Predictive Modelling**
  - YouTube Views Prediction
""", unsafe_allow_html=True)
