# Chat with Website - Langchain chatbot with Streamlit GUI
# " https://ethereum.org/en/nft/"
# Feature 

- **Website Interaction** : The Chatbot use the latest version of Langchain to interect with and extract inforamation from the various websites.

- **Large Language Model Integraction** : Compatibility with models like GPT 4, Mistral , Llama2. In this code I am using GPT - 3.5,  but you can change it yo amy other model.

- **Streamlit GUI**: A clean and intitive user interface built with streamlit,making it accessible for users with varying levels of technical expertise.

- **Python Based** :Entirely coded into Python.

# Installation 

Ensure you have to use Pyhton Installed on you system. Then clone this repository.

```bash 

git clone [repositry-link]
cd [repo - directory]

```

Install the required package :

```bash
pip insall -r requirements.txt 
```

create your own .env file with following variables :
```bash
OPEN_API_KEY = ["Your OpenAI API KEY"]
```

```bash
ACTIVELOOP_TOKEN = ["Your Activeloop Token"]
```

# usage 
To run the streamlit app : 

```bash
streamlit run chat-website.py
``` 

