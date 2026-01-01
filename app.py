import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and data
try:
    with open('random_forest_model.pkl', 'rb') as f:
        model_artifacts = pickle.load(f)
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
except FileNotFoundError:
    st.error("Model file 'random_forest_model.pkl' not found. Please train the model first.")
    st.stop()

try:
    df = pd.read_csv('California_Housing.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
except FileNotFoundError:
    st.error("Dataset 'California_Housing.csv' not found.")
    st.stop()

# Split data into train and test sets (same as training)
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    shuffle=True
)

# Compute predictions on test set only
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
y_true = y_test

# Calculate test metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
residuals = y_true - y_pred
absolute_errors = np.abs(residuals)
percentage_errors = (absolute_errors / y_true) * 100

# Feature names
feature_names = [col for col in df.columns if col != 'target']
feature_importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Header
st.title("California Housing Price Prediction System")
st.write("Random Forest Regressor Model for Housing Price Analysis and Prediction")
st.divider()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Data Exploration", "Model Performance", "Make Predictions", "Feature Analysis"]
)

# Display model information in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write("Model Type : Random Forest Regressor")
st.sidebar.write("Number of Trees : 300")
st.sidebar.write("Max Depth : 30")

# ==================== HOME PAGE ====================
if page == "Home":

    st.subheader("About This Application")
    st.write("This application uses a Random Forest Regressor to predict housing prices in California based on various features such as median income, house age, location, and more.")
    
    st.markdown("---")
    
    # Quick statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    
    with col2:
        st.metric("Features", len(df.columns) - 1)
    
    with col3:
        st.metric("Average Price", f"${df['target'].mean():.2f}00k")
    
    with col4:
        st.metric("Model Trees", model.n_estimators)
    
    st.markdown("---")
    
    # Dataset overview
    st.subheader("Dataset Overview")
    
    st.dataframe(df.head(10), width=800)
    
    
    # Feature descriptions
    st.markdown("---")
    st.subheader("Feature Descriptions")
    
    feature_descriptions = pd.DataFrame({
        'Feature': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'target'],
        'Description': [
            'Median income in block group (in tens of thousands)',
            'Median house age in block group',
            'Average number of rooms per household',
            'Average number of bedrooms per household',
            'Block group population',
            'Average number of household members',
            'Block group latitude',
            'Block group longitude',
            'Median house value (in hundreds of thousands of dollars)'
        ]
    })
    st.table(feature_descriptions)

# ==================== DATA EXPLORATION PAGE ====================
elif page == "Data Exploration":
    st.header("Data Exploration and Analysis")
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
    
    st.markdown("---")
    
    # Distribution plots
    st.subheader("Feature Distributions")
    
    tab1, tab2, tab3 = st.tabs(["Histograms", "Box Plots", "Correlation Analysis"])
    
    with tab1:
        st.write("Select features to visualize their distributions")
        selected_features = st.multiselect(
            "Select features:",
            df.columns.tolist(),
            default=['MedInc', 'HouseAge', 'target']
        )
        
        if selected_features:
            num_cols = min(3, len(selected_features))
            num_rows = (len(selected_features) + num_cols - 1) // num_cols
            
            fig = make_subplots(
                rows=num_rows, 
                cols=num_cols,
                subplot_titles=selected_features
            )
            
            for idx, feature in enumerate(selected_features):
                row = idx // num_cols + 1
                col = idx % num_cols + 1
                
                fig.add_trace(
                    go.Histogram(
                        x=df[feature],
                        name=feature,
                        nbinsx=50,
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(height=300*num_rows, showlegend=False)
            st.plotly_chart(fig)
    
    with tab2:
        st.write("Box plots show the distribution and outliers for each feature")
        
        selected_feature = st.selectbox(
            "Select a feature for detailed box plot:",
            df.columns.tolist()
        )
        
        fig = go.Figure()
        fig.add_trace(go.Box(y=df[selected_feature], name=selected_feature))
        fig.update_layout(
            title=f"Box Plot: {selected_feature}",
            yaxis_title=selected_feature,
            height=500
        )
        st.plotly_chart(fig)
        
        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{df[selected_feature].mean():.4f}")
        with col2:
            st.metric("Median", f"{df[selected_feature].median():.4f}")
        with col3:
            st.metric("Std Dev", f"{df[selected_feature].std():.4f}")
        with col4:
            st.metric("Range", f"{df[selected_feature].max() - df[selected_feature].min():.4f}")
    
    with tab3:
        st.write("Correlation matrix shows relationships between features")
        
        corr_matrix = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=600,
            width=700
        )
        st.plotly_chart(fig)
        
        # Show strongest correlations with target
        st.subheader("Strongest Correlations with Target")
        target_corr = corr_matrix['target'].drop('target').sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Positive Correlations:")
            positive_corr = target_corr[target_corr > 0]
            for feature, value in positive_corr.items():
                st.write(f"{feature}: {value:.4f}")
        
        with col2:
            st.write("Negative Correlations:")
            negative_corr = target_corr[target_corr < 0]
            for feature, value in negative_corr.items():
                st.write(f"{feature}: {value:.4f}")
    
    st.markdown("---")
    
    # Geographic visualization
    st.subheader("Geographic Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sample_df = df.sample(min(5000, len(df)))
        fig = px.scatter(
            sample_df,
            x='Longitude',
            y='Latitude',
            color='target',
            size='Population',
            hover_data=['MedInc', 'HouseAge'],
            title='Housing Prices by Location',
            color_continuous_scale='Viridis',
            labels={'target': 'Price (100k)'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig)
    
    with col2:
        st.write("Geographic Insights:")
        st.write(f"Latitude range: {df['Latitude'].min():.2f} to {df['Latitude'].max():.2f}")
        st.write(f"Longitude range: {df['Longitude'].min():.2f} to {df['Longitude'].max():.2f}")
        st.write("")
        st.write("Highest price area:")
        max_price_idx = df['target'].idxmax()
        st.write(f"Latitude: {df.loc[max_price_idx, 'Latitude']:.2f}")
        st.write(f"Longitude: {df.loc[max_price_idx, 'Longitude']:.2f}")
        st.write(f"Price: ${df.loc[max_price_idx, 'target']:.3f}00k")

# ==================== MODEL PERFORMANCE PAGE ====================
elif page == "Model Performance":
    st.header("Model Performance Analysis - Test Set")
    st.write(f"Performance metrics calculated on {len(y_test)} test samples (20% of total data)")
    
    # Display metrics
    st.subheader("Test Set Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
        st.caption("Variance Explained")
    
    with col2:
        st.metric("RMSE", f"{rmse:.4f}")
        st.caption("Root Mean Squared Error")
    
    with col3:
        st.metric("MAE", f"{mae:.4f}")
        st.caption("Mean Absolute Error")
    
    with col4:
        st.metric("MSE", f"{mse:.4f}")
        st.caption("Mean Squared Error")
    
    st.markdown("---")
    
    # Prediction vs Actual
    st.subheader("Prediction Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Prediction Scatter", "Residual Analysis", "Error Distribution"])
    
    with tab1:
        # Scatter plot of predictions vs actual
        sample_size = min(5000, len(y_test))
        if len(y_test) > sample_size:
            sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
        else:
            sample_indices = np.arange(len(y_test))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_true.iloc[sample_indices],
            y=y_pred[sample_indices],
            mode='markers',
            name='Predictions',
            marker=dict(
                size=5,
                color=y_pred[sample_indices],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Predicted Value"),
                opacity=0.6
            ),
            text=[f"Actual: {a:.3f}<br>Predicted: {p:.3f}" 
                  for a, p in zip(y_true.iloc[sample_indices], y_pred[sample_indices])],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"Predicted vs Actual Values (Sample of {sample_size})",
            xaxis_title="Actual Price (100k)",
            yaxis_title="Predicted Price (100k)",
            height=600,
            hovermode='closest'
        )
        
        st.plotly_chart(fig)
        
        # Show R² interpretation
        r2_pct = r2 * 100
        st.info(f"The R² score of {r2:.4f} indicates that the model explains {r2_pct:.2f}% of the variance in housing prices.")
    
    with tab2:
        # Residual plot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Residuals vs Predicted Values", "Residuals Distribution")
        )
        
        # Residuals scatter
        fig.add_trace(
            go.Scatter(
                x=y_pred[sample_indices],
                y=residuals.iloc[sample_indices],
                mode='markers',
                marker=dict(size=5, opacity=0.6),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Zero line
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Residuals histogram
        fig.add_trace(
            go.Histogram(
                y=residuals,
                nbinsy=50,
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        
        fig.update_layout(height=500)
        st.plotly_chart(fig)
        
        # Residual statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Residual", f"{residuals.mean():.6f}")
        with col2:
            st.metric("Std Dev of Residuals", f"{residuals.std():.4f}")
        with col3:
            st.metric("Median Residual", f"{residuals.median():.6f}")
    
    with tab3:
        # Error distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=absolute_errors,
                nbinsx=50,
                name='Absolute Errors'
            ))
            fig.update_layout(
                title="Distribution of Absolute Errors",
                xaxis_title="Absolute Error",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig)
            
            # Error statistics
            st.write("Absolute Error Statistics:")
            st.write(f"Mean: {absolute_errors.mean():.4f}")
            st.write(f"Median: {absolute_errors.median():.4f}")
            st.write(f"75th percentile: {absolute_errors.quantile(0.75):.4f}")
            st.write(f"95th percentile: {absolute_errors.quantile(0.95):.4f}")
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=percentage_errors,
                nbinsx=50,
                name='Percentage Errors'
            ))
            fig.update_layout(
                title="Distribution of Percentage Errors",
                xaxis_title="Percentage Error (%)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig)
            
            # Percentage error statistics
            st.write("Percentage Error Statistics:")
            st.write(f"Mean: {percentage_errors.mean():.2f}%")
            st.write(f"Median: {percentage_errors.median():.2f}%")
            st.write(f"75th percentile: {percentage_errors.quantile(0.75):.2f}%")
            st.write(f"95th percentile: {percentage_errors.quantile(0.95):.2f}%")
        
        # Accuracy by percentage threshold
        st.markdown("---")
        st.subheader("Prediction Accuracy by Threshold")
        
        thresholds = [5, 10, 15, 20, 25]
        accuracies = [(percentage_errors <= t).sum() / len(percentage_errors) * 100 for t in thresholds]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Within {t}%" for t in thresholds],
            y=accuracies,
            text=[f"{a:.1f}%" for a in accuracies],
            textposition='outside',
            marker_color='lightblue'
        ))
        fig.update_layout(
            title="Percentage of Predictions within Error Threshold",
            xaxis_title="Error Threshold",
            yaxis_title="Percentage of Predictions (%)",
            height=400
        )
        st.plotly_chart(fig)

# ==================== MAKE PREDICTIONS PAGE ====================
elif page == "Make Predictions":
    st.header("Make Housing Price Predictions")
    
    st.write("Enter the property details below to get a price prediction")
    
    # Two modes: manual input and sample selection
    prediction_mode = st.radio(
        "Select prediction mode:",
        ["Manual Input", "Select from Dataset"]
    )
    
    if prediction_mode == "Manual Input":
        # Manual input form
        st.subheader("Enter Property Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            med_inc = st.number_input(
                "Median Income (tens of thousands)",
                min_value=0.0,
                max_value=15.0,
                value=3.5,
                step=0.1,
                help="Median income in the area in tens of thousands of dollars"
            )
            
            house_age = st.number_input(
                "House Age (years)",
                min_value=1,
                max_value=52,
                value=25,
                step=1,
                help="Median age of houses in the block group"
            )
            
            ave_rooms = st.number_input(
                "Average Rooms",
                min_value=1.0,
                max_value=20.0,
                value=5.5,
                step=0.1,
                help="Average number of rooms per household"
            )
            
            ave_bedrms = st.number_input(
                "Average Bedrooms",
                min_value=0.5,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Average number of bedrooms per household"
            )
        
        with col2:
            population = st.number_input(
                "Population",
                min_value=1.0,
                max_value=10000.0,
                value=1500.0,
                step=10.0,
                help="Total population in the block group"
            )
            
            ave_occup = st.number_input(
                "Average Occupancy",
                min_value=0.5,
                max_value=20.0,
                value=3.0,
                step=0.1,
                help="Average number of household members"
            )
            
            latitude = st.number_input(
                "Latitude",
                min_value=32.0,
                max_value=42.0,
                value=37.0,
                step=0.01,
                help="Block group latitude"
            )
            
            longitude = st.number_input(
                "Longitude",
                min_value=-125.0,
                max_value=-114.0,
                value=-120.0,
                step=0.01,
                help="Block group longitude"
            )
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'MedInc': [med_inc],
            'HouseAge': [house_age],
            'AveRooms': [ave_rooms],
            'AveBedrms': [ave_bedrms],
            'Population': [population],
            'AveOccup': [ave_occup],
            'Latitude': [latitude],
            'Longitude': [longitude]
        })
        
        actual_price_available = False
        
    else:
        # Sample selection mode
        st.subheader("Select a Sample from Dataset")
        
        sample_idx = st.number_input(
            "Enter sample index (0 to {})".format(len(df)-1),
            min_value=0,
            max_value=len(df)-1,
            value=0,
            step=1
        )
        
        sample = df.iloc[sample_idx].drop('target')
        input_data = pd.DataFrame([sample])
        
        st.write("Selected Sample Details:")
        st.dataframe(input_data)
        
        actual_price = df.iloc[sample_idx]['target']
        st.write(f"Actual Price: ${actual_price:.3f}00k")
        actual_price_available = True
    
    # Prediction button
    if st.button("Predict Price", type="primary"):
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display prediction
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.success("Predicted Price")
            st.title(f"${prediction:.3f}00k")
            st.write(f"or approximately ${prediction * 100:.0f},000")
        
        # Show prediction details
        st.markdown("---")
        st.subheader("Prediction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Input Features:")
            for col in input_data.columns:
                st.write(f"{col}: {input_data[col].values[0]:.4f}")
        
        with col2:
            st.write("Prediction Statistics:")
            st.write(f"Prediction: ${prediction:.3f}00k")
            st.write(f"Dataset Mean: ${df['target'].mean():.3f}00k")
            st.write(f"Dataset Median: ${df['target'].median():.3f}00k")
            
            diff_from_mean = ((prediction - df['target'].mean()) / df['target'].mean()) * 100
            st.write(f"Difference from Mean: {diff_from_mean:+.2f}%")
            
            if actual_price_available:
                st.write(f"Actual Price: ${actual_price:.3f}00k")
                error = abs(prediction - actual_price)
                error_pct = (error / actual_price) * 100
                st.write(f"Prediction Error: ${error:.3f}00k ({error_pct:.2f}%)")
        
        # Confidence interval estimate
        predictions_all_trees = np.array([tree.predict(input_scaled)[0] for tree in model.estimators_])
        pred_std = predictions_all_trees.std()
        
        st.markdown("---")
        st.subheader("Prediction Confidence")
        
        confidence_interval = 1.96 * pred_std
        
        st.write(f"Standard Deviation: {pred_std:.4f}")
        st.write(f"95% Confidence Interval: ${prediction - confidence_interval:.3f}00k  to  ${prediction + confidence_interval:.3f}00k")
        
        # Visualize prediction distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=predictions_all_trees,
            nbinsx=30,
            name='Tree Predictions'
        ))
        fig.add_vline(
            x=prediction,
            line_dash="dash",
            line_color="red",
            annotation_text="Final Prediction"
        )
        fig.update_layout(
            title="Distribution of Predictions Across All Trees",
            xaxis_title="Predicted Price (100k)",
            yaxis_title="Number of Trees",
            height=400
        )
        st.plotly_chart(fig)

# ==================== FEATURE ANALYSIS PAGE ====================
elif page == "Feature Analysis":
    st.header("Feature Importance Analysis")
    
    # Display feature importances
    st.subheader("Feature Importance Rankings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_importances_df['Importance'],
            y=feature_importances_df['Feature'],
            orientation='h',
            marker_color='lightblue',
            text=feature_importances_df['Importance'].apply(lambda x: f'{x:.4f}'),
            textposition='outside'
        ))
        fig.update_layout(
            title="Feature Importance Scores",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig)
    
    with col2:
        st.write("Feature Rankings:")
        for idx, row in feature_importances_df.iterrows():
            percentage = row['Importance'] * 100
            st.write(f"{row['Feature']}: {percentage:.2f}%")
        
        st.markdown("---")
        st.write("Top 3 Features:")
        top_3 = feature_importances_df.head(3)
        for idx, row in top_3.iterrows():
            st.success(f"{row['Feature']}: {row['Importance']:.4f}")
    
    # Feature importance pie chart
    st.markdown("---")
    st.subheader("Feature Contribution Breakdown")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure(data=[go.Pie(
            labels=feature_importances_df['Feature'],
            values=feature_importances_df['Importance'],
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        fig.update_layout(
            title="Proportion of Each Feature's Importance",
            height=500
        )
        st.plotly_chart(fig)
    
    with col2:
        st.write("Cumulative Importance:")
        cumulative = feature_importances_df['Importance'].cumsum()
        for idx, (feat_idx, row) in enumerate(feature_importances_df.iterrows()):
            st.write(f"Top {idx+1}: {cumulative.iloc[idx]*100:.2f}%")
    
    # Feature relationships with target
    st.markdown("---")
    st.subheader("Feature Relationships with Target")
    
    selected_feature = st.selectbox(
        "Select a feature to analyze:",
        feature_names
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot
        sample_size = min(5000, len(df))
        sample_df = df.sample(sample_size)
        
        fig = px.scatter(
            sample_df,
            x=selected_feature,
            y='target',
            opacity=0.5,
            trendline='lowess',
            title=f'{selected_feature} vs Target Price',
            labels={'target': 'Price (100k)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig)
    
    with col2:
        # Binned analysis
        st.write(f"{selected_feature} Binned Analysis:")
        
        # Create bins
        bins = pd.qcut(df[selected_feature], q=5, duplicates='drop')
        binned_stats = df.groupby(bins)['target'].agg(['mean', 'median', 'count'])
        
        st.dataframe(binned_stats)
        
        # Show correlation
        correlation = df[selected_feature].corr(df['target'])
        st.metric("Correlation with Target", f"{correlation:.4f}")
    
    # Feature impact analysis
    st.markdown("---")
    st.subheader("Feature Impact on Predictions")
    st.write("This shows how changing one feature affects predictions (holding others at median)")
    
    feature_to_analyze = st.selectbox(
        "Select a feature for impact analysis:",
        feature_names,
        key='impact_analysis'
    )
    
    # Get median values for all features
    median_values = df[feature_names].median()
    
    # Create range for selected feature
    feature_min = df[feature_to_analyze].quantile(0.05)
    feature_max = df[feature_to_analyze].quantile(0.95)
    feature_range = np.linspace(feature_min, feature_max, 50)
    
    # Make predictions across range
    predictions = []
    for val in feature_range:
        input_data_temp = median_values.copy()
        input_data_temp[feature_to_analyze] = val
        input_df_temp = pd.DataFrame([input_data_temp])
        input_scaled_temp = scaler.transform(input_df_temp)
        pred = model.predict(input_scaled_temp)[0]
        predictions.append(pred)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=feature_range,
        y=predictions,
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='blue', width=3)
    ))
    fig.update_layout(
        title=f'Impact of {feature_to_analyze} on Predicted Price',
        xaxis_title=feature_to_analyze,
        yaxis_title='Predicted Price (100k)',
        height=500
    )
    st.plotly_chart(fig)
    
    # Show insights
    price_change = predictions[-1] - predictions[0]
    price_change_pct = (price_change / predictions[0]) * 100
    
    st.info(f"Changing {feature_to_analyze} from {feature_min:.2f} to {feature_max:.2f} changes predicted price by ${price_change:.3f}00k ({price_change_pct:+.2f}%)")


def about_the_coder():
    # We use a non-indented string to prevent Markdown from treating it as code
    html_code = """
    <style>
    .coder-card {
        background-color: transparent;
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 10px;
        padding: 20px;
        display: flex;
        align-items: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .coder-img {
        width: 100px; /* Slightly larger for better visibility */
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #FF4B4B; /* Streamlit Red */
        margin-right: 25px;
        flex-shrink: 0; /* Prevents image from shrinking */
    }
    .coder-info h3 {
        margin: 0;
        font-family: 'Source Sans Pro', sans-serif;
        color: inherit;
        font-size: 1.4rem;
        font-weight: 600;
    }
    .coder-info p {
        margin: 10px 0;
        font-size: 1rem;
        opacity: 0.9;
        line-height: 1.5;
    }
    .social-links {
        margin-top: 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
    }
    .social-links a {
        text-decoration: none;
        color: #FF4B4B;
        font-weight: bold;
        font-size: 0.95rem;
        transition: color 0.3s;
    }
    .social-links a:hover {
        color: #ff2b2b;
        text-decoration: underline;
    }
    /* Mobile responsiveness */
    @media (max-width: 600px) {
        .coder-card {
            flex-direction: column;
            text-align: center;
            padding: 15px;
        }
        .coder-img {
            margin-right: 0;
            margin-bottom: 15px;
            width: 80px;
            height: 80px;
        }
        .social-links {
            justify-content: center;
        }
    }
    </style>  
    <div class="coder-card">
        <img src="https://ui-avatars.com/api/?name=Yash+Vasudeva&size=120&background=FF4B4B&color=fff&bold=true&rounded=true" class="coder-img" alt="Yash Vasudeva"/>
        <div class="coder-info">
            <h3>Developed by Yash Vasudeva</h3>
            <p>
                Results-driven Data & AI Professional skilled in <b>Data Analytics</b>, 
                <b>Machine Learning</b>, and <b>Deep Learning</b>. 
                Passionate about transforming raw data into business value and building intelligent solutions.
            </p>
            <div class="social-links">
                <a href="https://www.linkedin.com/in/yash-vasudeva/" target="_blank">LinkedIn</a>
                <a href="https://github.com/yashvasudeva1" target="_blank">GitHub</a>
                <a href="mailto:vasudevyash@gmail.com">Contact</a>
                <a href="https://yashvasudeva.vercel.app/" target="_blank">Portfolio</a>
            </div>
        </div>
    </div>
    """
        
    st.markdown(html_code, unsafe_allow_html=True)

st.divider()

if __name__ == "__main__":
    about_the_coder()
