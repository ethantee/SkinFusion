# Part 1: Initial setup and loading resources
# -------------------------------------------
# Import the necessary libraries
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import time

from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader
from urllib.parse import urlparse, urlunparse # for cleaning image URLs

# Configurations
st.set_page_config(
    layout="wide",
    page_icon="🌿",
    page_title="Hybrid Skincare Product Recommendation System",
)

# Author preferences and default configurations
DEFAULT_AUTHOR_ID = "21684758896"
DEFAULT_MAX_BUDGET = "200"
DEFAULT_RECOMMENDATION_COUNT = 5
MIN_RECOMMENDATIONS = 3
MAX_RECOMMENDATIONS = 20

# Google API key and search engine ID
API_KEY = "-"           # Google Custom Search API Key
SEARCH_ENGINE_ID = "-"  # Google Custom Search Engine ID

# Initialize progress bar and display message
progress_text = st.empty()
progress_text.markdown("##### Starting to load resources...")
progress_bar = st.progress(0)

# Function to load all resources
@st.cache_resource(show_spinner=False)
def load_resources():
    with st.spinner("Loading Product Information..."):
        # Load the product information dataset
        product_info = pd.read_csv("data/product_info.csv")
        product_info["product_id"] = product_info["product_id"].astype(str)
        progress_bar.progress(0.1)
        time.sleep(0.2)
    
    with st.spinner("Loading Cosine Similarity Matrix (ETA ≈ 20s) ..."):
        # Load the cosine similarity matrix
        cosine_sim_matrix = np.load("deployment/cosine_sim_matrix_glove.npy")
        progress_bar.progress(0.5)
        time.sleep(0.2)
    
    with st.spinner("Loading Best SVD Model..."):
        # Load the best SVD model
        best_svd_model = joblib.load("deployment/optimized_svd_model.pkl")
        progress_bar.progress(0.6)
        time.sleep(0.2)
    
    with st.spinner("Loading Datasets..."):
        # Load the datasets
        skincare_reviews_cbf = pd.read_feather("processed_data/skincare_reviews_cbf.feather")
        progress_bar.progress(0.8)
        time.sleep(0.2)
        skincare_reviews_cf = pd.read_feather("processed_data/skincare_reviews_cf.feather")
        progress_bar.progress(1.0)
        time.sleep(0.5)
        
    # Prepare the trainset for the SVD model
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(skincare_reviews_cf[["author_id", "product_id", "calibrated_author_rating"]], reader)
    trainset = data.build_full_trainset()
    
    return product_info, cosine_sim_matrix, best_svd_model, skincare_reviews_cbf, skincare_reviews_cf, trainset

# Load all resources and store in session states
if "resources_loaded" not in st.session_state:
    resources = load_resources()
    st.session_state["product_info"], \
    st.session_state["cosine_sim_matrix"], \
    st.session_state["best_svd_model"], \
    st.session_state["skincare_reviews_cbf"], \
    st.session_state["skincare_reviews_cf"], \
    st.session_state["svd_trainset"] = resources
    st.session_state["resources_loaded"] = True

# Remove progress bar and text
progress_text.text("All resources loaded!")
progress_text.empty()
progress_bar.empty()

# Part 2: Define all necessary functions
# --------------------------------------
# Prepare a deduplicated product details DataFrame for efficient merging
product_details = st.session_state["skincare_reviews_cf"].drop_duplicates(subset="product_id")[["product_id",
                                                                                                "product_name",
                                                                                                "brand_name",
                                                                                                "price_usd"]]
product_price = product_details.set_index("product_id")["price_usd"].to_dict()

# Function to perform content-based filtering recommendations using GloVe embeddings
def get_cbf_glove_recommendations(author_id, top_n, max_budget):
    # Retrieve all reviews by the specified author from the skincare_reviews dataset
    author_reviews = st.session_state["skincare_reviews_cbf"].query("author_id == @author_id")
    
    # Return error message if no reviews are found for the author
    if author_reviews.empty:
        return f"Author {author_id} is not found! Please try another author."
    
    # Calculate the author's average profile from their reviewed product feature vectors
    author_profile = np.mean(np.stack(author_reviews["vectorized_features_glove"].values), axis=0)

    # Select products not reviewed by the author and within the specified budget
    filtered_products = st.session_state["skincare_reviews_cbf"].loc[
        (~st.session_state["skincare_reviews_cbf"]["product_id"].isin(author_reviews["product_id"])) &
        (st.session_state["skincare_reviews_cbf"]["price_usd"] <= max_budget)
    ].copy()
    
    # Return error message if no products are found within the budget
    if filtered_products.empty:
        return f"No products found within budget of ${max_budget}. Try increasing the budget."
    
    # Compute similarity scores
    filtered_products["similarity_score"] = cosine_similarity(
        [author_profile],
        np.stack(filtered_products["vectorized_features_glove"].values)
    )[0]

    # Get the top N recommendations
    final_recommendations = filtered_products.query("similarity_score < 1").sort_values(
        by="similarity_score", ascending=False
    ).drop_duplicates(subset="product_id").head(top_n)[
        ["product_id", "product_name", "brand_name", "price_usd", "average_rating", "similarity_score"]
    ]
    
    return final_recommendations.reset_index(drop=True)

# Function to perform collaborative filtering recommendations using optimized SVD model
def get_cf_svd_recommendations(author_id, best_svd_model, top_n, max_budget):
    # Check if the author exists in the trainset
    try:
        inner_author_id = st.session_state["svd_trainset"].to_inner_uid(author_id)
    except ValueError:
        return f"Author {author_id} is not found! Please try another author."
    
    # Get all products and the products rated by the author
    all_products = set(st.session_state["skincare_reviews_cf"]["product_id"].unique())
    rated_products = set(st.session_state["skincare_reviews_cf"].query("author_id == @author_id and calibrated_author_rating > 0")["product_id"].unique())
    
    # Determine unrated products
    unrated_products = all_products - rated_products

    # Pre-filter products within the budget to reduce unnecessary predictions
    affordable_unrated_products = {prod for prod in unrated_products if product_price.get(prod, float("inf")) <= max_budget}

    # Predict ratings for the affordable unrated items
    predictions = [best_svd_model.predict(author_id, prod, clip=True) for prod in affordable_unrated_products]
    
    # Check if there are any predictions
    if not predictions:
        return f"No products found within budget of ${max_budget}. Try increasing the budget."

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Build the final recommendations DataFrame
    top_recs = pd.DataFrame([(pred.iid, pred.est) for pred in predictions[:top_n]], columns=["product_id", "predicted_rating"])
    
    # Merge with product details
    final_recommendations = top_recs.merge(product_details, on="product_id", how="left").head(top_n)
    
    # Reset the index and return the final recommendations
    return final_recommendations[["product_id", "product_name", "brand_name", "price_usd", "predicted_rating"]].reset_index(drop=True)

# Function to perform hybrid recommendations using a fixed ratio of CBF to CF recommendations (2:1 ratio by default)
def hybrid_ratio_recommendations(cbf_recs, cf_recs, cbf_to_cf_ratio, top_n):
    hybrid_recs = []
    cb_index, cf_index = 0, 0
    total_cbf = len(cbf_recs)
    total_cf = len(cf_recs)
    cb_quota, cf_quota = cbf_to_cf_ratio
    
    # Loop until the top N recommendations are reached or all recommendations are exhausted
    while len(hybrid_recs) < top_n and (cb_index < total_cbf or cf_index < total_cf):
        # Add content-based filtering recommendations according to the ratio
        for _ in range(cb_quota):
            if cb_index < total_cbf and len(hybrid_recs) < top_n:
                hybrid_recs.append(cbf_recs.iloc[cb_index])
                cb_index += 1

        # Add collaborative filtering recommendations according to the ratio
        for _ in range(cf_quota):
            if cf_index < total_cf and len(hybrid_recs) < top_n:
                hybrid_recs.append(cf_recs.iloc[cf_index])
                cf_index += 1

    # Convert list to DataFrame
    return pd.DataFrame(hybrid_recs).reset_index(drop=True)

# Function to get products reviewed by the author with sorting
def get_author_reviewed_products(author_id):
    reviewed_products = st.session_state["skincare_reviews_cbf"].query(
        "author_id == @author_id"
    )[["product_id", "product_name", "brand_name", "price_usd"]]
    
    # Sort the DataFrame first by "brand_name" and then by "product_name"
    reviewed_products_sorted = reviewed_products.sort_values(by=["brand_name", "product_name"])
    
    # Start index from 1
    reviewed_products_sorted.index = range(1, len(reviewed_products_sorted) + 1)
    
    return reviewed_products_sorted

# Function to perform recommendation validation
def validate_recommendation_input(author_id, max_budget, author_values):
    error_messages = []
    try:
        max_budget = float(max_budget)
        if max_budget <= 0:
            error_messages.append("a positive maximum budget in dollars")
    except ValueError:
        error_messages.append("a numeric value for maximum budget in dollars")
    
    if author_id not in author_values:
        error_messages.append("a valid Author ID")
    
    return error_messages, max_budget

# Function to clean the image URL by removing query parameters
def clean_image_url(url):
    parsed_url = urlparse(url)
    clean_url = urlunparse(parsed_url._replace(query="")) # reconstruct the URL without query parameters
    
    return clean_url

# Function to fetch images using the Google Custom Search API
def fetch_images(query, num_images=1):
    # URL endpoint for Google Custom Search JSON API
    url = "https://www.googleapis.com/customsearch/v1"
    
    # Parameters for the API request
    params = {
        "q": query,             # search query
        "cx": SEARCH_ENGINE_ID, # search engine ID
        "key": API_KEY,         # API key
        "searchType": "image",  # search for images
        "num": num_images,      # number of images
    }
    
    # Send a GET request to the API
    response = requests.get(url, params=params)
    results = response.json() # parse the JSON response
    
    # Extract image URLs from the results
    if "items" in results:
        return [clean_image_url(item["link"]) for item in results["items"]]
    
    return [] # return an empty list if no images are found

# Part 3: Create the Streamlit application
# ----------------------------------------
# Title of my FYP application
st.title("🌿Hybrid Skincare Product Recommendation System🌿")
st.write("Welcome to the Hybrid Skincare Product Recommendation System! "
         "Just enter your **Author ID**, **preferred budget**, and **desired number of product recommendations**. "
         "The advanced hybrid algorithm seamlessly integrates content-based and collaborative filtering "
         "to curate a personalized list just for you.")

# Generate 5 random Author IDs for demonstration purposes
if "resources_loaded" in st.session_state:
    random_author_ids = st.session_state["skincare_reviews_cf"]["author_id"].sample(5).tolist()

# Sidebar for author inputs
with st.sidebar:
    # Display the SkinFusion logo
    st.image("images/skin_fusion.png", use_column_width=True)
    
    # Add a title to the sidebar
    st.header("Author Preferences")
    
    # Add text inputs for author preferences
    author_id = (st.text_input("Author ID", DEFAULT_AUTHOR_ID))        
    max_budget = st.text_input("Maximum Budget ($)", DEFAULT_MAX_BUDGET)
    top_n = st.slider("Number of Recommendations", MIN_RECOMMENDATIONS,
                                                   MAX_RECOMMENDATIONS,
                                                   DEFAULT_RECOMMENDATION_COUNT)
    
    # Validate the author_id
    try:
        author_id = int(author_id)
        invalid_author_id = False
    except ValueError:
        author_id = None
        invalid_author_id = True
    
    # Button to trigger recommendations
    recommend_button = st.button("Get Personalized Recommendations")
    st.write("---") # divider
    
    # Button to generate random author IDs
    generate_button = st.button("Generate Random Author IDs")
    if generate_button:
        random_author_ids = st.session_state["skincare_reviews_cf"]["author_id"].sample(5).tolist()
        author_ids_formatted = "\n".join(f"{index + 1}. {id}" for index, id in enumerate(random_author_ids))
        st.session_state["author_ids_list"] = author_ids_formatted
    
    # Display the random author IDs
    if "author_ids_list" in st.session_state:
        st.write(st.session_state["author_ids_list"])
    
# Main page to display recommendations
if recommend_button and not generate_button:
    # Validate the input
    errors, max_budget = validate_recommendation_input(author_id,
                                                       max_budget,
                                                       st.session_state["skincare_reviews_cf"]["author_id"].values)
    
    # Display error messages (if any)
    if errors:
        error_message = " and ".join(errors)
        st.error(f"Please enter {error_message}.")
        st.stop()
    
    # Generate recommendations
    with st.spinner("Generating Recommendations..."):
        cbf_recommendations = get_cbf_glove_recommendations(author_id, top_n, max_budget)
        if isinstance(cbf_recommendations, str):
            st.error(cbf_recommendations)
            st.stop()
            
        cf_recommendations = get_cf_svd_recommendations(author_id, st.session_state["best_svd_model"], top_n, max_budget)
        if isinstance(cf_recommendations, str):
            st.error(cf_recommendations)
            st.stop()
    
    # Perform hybrid recommendations
    hybrid_recommendations = hybrid_ratio_recommendations(cbf_recommendations, cf_recommendations, (2, 1), top_n)
    
    # Store the results in session state
    st.session_state["hybrid_recommendations"] = hybrid_recommendations
    
    # Validate successful recommendations
    st.success("Recommendations Generated Successfully!")
    
    # Merge additional product information with the hybrid recommendations
    detailed_recommendations = pd.merge(
        hybrid_recommendations,
        st.session_state["product_info"][["product_id", "loves_count", "rating", "secondary_category",
                                          "tertiary_category", "size", "variation_type", "highlights"]],
        on="product_id",
        how="left"
    )
    
    # Display the recommendations
    if "hybrid_recommendations" in st.session_state:
        # Display the title for the recommendations
        st.subheader(f"Top {top_n} Hybrid Recommendations for Author {author_id}:")
        
        # Select columns to display
        displayed_columns = ["product_id", "product_name", "brand_name", "price_usd"]
        
        # Start index from 1
        hybrid_recommendations.index += 1
        
        # Display the hybrid recommendations results
        st.table(hybrid_recommendations[displayed_columns])
        st.markdown("**The recommendations above are weighted based on (2:1 CBF to CF Ratio).*") 
        st.write("---")
        
        # Create expanders for each recommendation with detailed information
        for _, row in detailed_recommendations.iterrows():
            # Convert the highlights column to a list
            if isinstance(row["highlights"], str):
                try:
                    # Convert a string representation of a list into an actual list
                    highlights_list = eval(row["highlights"])
                except:
                    # Wrap the string in a list if it is not a valid list
                    highlights_list = [row["highlights"]]
            elif row["highlights"] is None:
                highlights_list = []  # set to empty list if None
            else:
                # Assume highlights are already a list
                highlights_list = row["highlights"]
            
            # Convert the highlights list to a string for display
            if isinstance(highlights_list, list):
                highlights_str = ", ".join(highlights_list)
            else:
                highlights_str = str(highlights_list)  # convert to string if not a list
            
             # Define the product name for image search
            product_name = f"Sephora {row['product_name']}" # add "Sephora" for better image search results
            
            # Create an expander for each product
            with st.expander(f"[**{row['brand_name']}**] **{row['product_name']}**"):
                # Fetch the image for the product
                image_urls = fetch_images(product_name)
                if image_urls:
                    st.image(image_urls[0], width=400) # display the image with 400px width
                    
                # Display the remaining product details
                st.markdown(f"**Product ID:** {row['product_id']}")
                st.markdown(f"**Price:** ${row['price_usd']:.2f}")
                st.markdown(f"**Loves Count:** {row['loves_count']}")
                st.markdown(f"**Average Rating:** {row['rating']:.2f}")
                st.markdown(f"**Secondary Category:** {row['secondary_category']}")
                st.markdown(f"**Tertiary Category:** {row['tertiary_category']}")
                st.markdown(f"**Product Size:** {row['size']}")
                st.markdown(f"**Variation Type:** {row['variation_type']}")
                st.markdown(f"**Highlights:** {highlights_str}")
        
        # Display the author's reviewed products
        st.write("---")
        product_count = len(get_author_reviewed_products(author_id))
        st.markdown(f"#### Author {author_id} has reviewed the following {product_count} products:")
        st.table(get_author_reviewed_products(author_id))
