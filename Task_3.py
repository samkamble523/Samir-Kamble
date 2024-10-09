import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

responses = {
    "what is castle swimmer about": "Castle Swimmer is a webtoon that follows the story of a young mermaid prince navigating a world of magic, danger, and self-discovery.",
    "who are the main characters": "The main characters include the mermaid prince and his companions, who face various challenges and adventures in their quest.",
    "what genre is castle swimmer": "Castle Swimmer falls under the fantasy and adventure genres.",
    "tell me about the prophecy": "The prophecy in Castle Swimmer hints at significant events that will shape the destinies of the characters.",
    "how many chapters are there": "As of now, Castle Swimmer has numerous chapters, with Chapters 83-89 exploring new revelations.",
}

def get_response(user_input):
    # Tokenize the user input for processing
    tokens = word_tokenize(user_input.lower())
    
    # Check for keywords in the user's input and return the appropriate response
    for key in responses.keys():
        if key in user_input.lower():
            return responses[key]
    
    # If no keywords are found, return a default response
    return "I'm sorry, I don't have the answer to that."

def chat():
    print("Welcome to the Castle Swimmer Chatbot! (type 'exit' to stop)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input)
        print("Chatbot:", response)

chat()
