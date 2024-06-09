import streamlit as st
import requests
import json

# Function to check the token and get user data
def check_token(token):
    with open("./PacemakerInnovationsData/authentication.json", "r") as f:
        auth_data = json.load(f)
    user_data = next((user for user in auth_data if user["token"] == token), None)
    return user_data

# Load topic index
def load_topic_index():
    with open("./PacemakerInnovationsData/topicIndex.json", "r") as f:
        return json.load(f)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Login page
if not st.session_state.authenticated:
    st.title("Login")
    token = st.text_input("User Token", type="password")
    if st.button("Login"):
        user_data = check_token(token)
        if user_data:
            st.session_state.authenticated = True
            st.session_state.token = token
            st.session_state.user_data = user_data
            st.success("Login successful")
            st.experimental_rerun()  # Rerun the script to display the main application
        else:
            st.error("Invalid token")
else:
    # Main application
    st.title("Knowledge Graph Interface")

    # Load topic index
    topic_index = load_topic_index()

    # Section for adding knowledge
    st.header("Add Knowledge")
    summary = st.text_input("Summary")
    content = st.text_area("Content")
    authors = st.text_input("Authors (comma-separated)")

    # Dropdown for topics
    allowed_topics = [topic_index[str(i)] for i in st.session_state.user_data["topic_access_indices"]]
    selected_topic = st.selectbox("Select Topic", allowed_topics)

    if st.button("Add Knowledge"):
        if summary and content and authors and selected_topic:
            data = {
                "summary": summary,
                "content": content,
                "authors": authors.split(",")
            }
            response = requests.post("http://127.0.0.1:8000/add_knowledge", json=data, params={"token": st.session_state.token})
            if response.status_code == 200:
                st.success("Knowledge added successfully")
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        else:
            st.error("Please fill in all fields")

    # Section for querying knowledge
    st.header("Query Knowledge")
    query = st.text_input("Query")

    if st.button("Send Query"):
        if query:
            data = {"query": query}
            response = requests.post("http://127.0.0.1:8000/query", json=data, params={"token": st.session_state.token})
            if response.status_code == 200:
                st.write(response.json().get("answer", "No answer found"))
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        else:
            st.error("Please provide a query")