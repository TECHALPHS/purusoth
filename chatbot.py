import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter
    st.title("Intents of chatbot using NLP")

    # Create a sidebar menu with options
    menu = ["Home","Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the converstion")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User input', 'Chatbot Response', 'Timestamp'])
                
        counter += 1
        user_input = st.text_input("YOu:", key=f"user_input_{counter}")

        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            #Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, respone, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

     # Conversation History Menu
     elif choice == "Conversation History":
         # Display the conversation history in a collapsible expander
         st.header("Conversation History")
         # with st.beta_expander("Click to see Conversationn expander
         with open('chat_log.csv', 'r', encoding= 'utf-8') as csvfile:
             csv_reader = csv.reader(csvfile)
             next(csv_reader)  # Skip the header row
             for row in csv_reader:
                 st.text(f"User: {row[0]}")
                 st.text(f"Chatbot: {row[1]}")
                 st.text(f"Timestamp: {row[2]}")
                 st.markdown("---")
                 
     elif choice == "About":
         st.write("The goal of this project is to create a chatbot that can understand and respond to the user input")

         st.subheader("Dataset:")

         st.write("""
         The Project is divided into two parts:
         1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on label
         2. For building the Chatbot interface, Streamlit web framework is used to build a web-p
         """)

         st.subheader("Dataset:")
         
         st.write("""
         The dataset used inn this project is a collection of labelled intents and entities.
         - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
         - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget", "I am a chatbot")
         - Text: The user input text.
         """)

         st.subheader("Streamlit Chatbot Interface:")

         st.write("The chatbot interface is built using streamlit. The interface includes a text command")

         st.subheader("Conclusion:")

         st.write("In this project, a chatbot is built that can understand and respond to useer input")

if __name__ == '__main__':
    main()
                               

        
        
