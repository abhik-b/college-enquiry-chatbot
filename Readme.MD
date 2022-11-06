# College Enquiry Chatbot

## What does this College Enquiry Chatbot do ?

This chatbot will respond to students' queries regarding university. Questions like "When are holidays ? ", "Can you tell more about teachers?","How long does MCA take ?","What is my CGPA this sem" can be answered by this chatbot.

All features :

- Ask result or CGPA
- Queries related to faculty
- Course details & syllabus
- Ask about holidays


## Tech Stack
- **database** - sqlite3
- **server** - python & flask   
- **frontend** - html,css,js  
- **neural network** - python & pytorch
- **natural language processing** - spacy & nltk

## Getting started

First create a virtual environment using `virtualenv venv` command in **terminal/cmd prompt/powershell**. Then activate it. Now we can install our dependencies for this project inside this isolated virtual environment.

We will install the following libraries to get started :

- **torch** - to create the neural network
- **nltk & spacy** - for NLP (natural language processing)
- **flask** - to run the web server where
  - we will interact with our chatbot
  - admin dashboard to manage our teachers, students, courses

> I went to Pytorch's [ official guide ](https://pytorch.org/get-started/locally/)to start local developement & got the command I need to install pytorch with cuda.

I used `pip install nltk spacy flask` to install them together. Then I use `python -m spacy download en_core_web_sm` to download english core small version.

**Note** : I have used CUDA (GPU) so torch versions have `+cu116` written. When installing from _requirements.txt_ use this command :

`pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`

## Project structure

```
-- flask_server (this is the server)
        -- static (contains css & js files)
        -- templates (contains all html files)
        -- university
            -- __init__.py (to help importing university folder easily throughout the project)
            -- models.py (contains database models)
            -- routes.py (contains routes for university related views & apis)
        -- __init__.py (initializing base flask app throughout the project)
        -- university.db
-- chat.py
-- neural_net.py (base model for neural net)
-- nlp_utils.py
-- train.py (run this file to train our neural net)
-- run.py ( this file will run our flask sever)
-- intents.json (dataset for training)
-- requirement.txt (stores the name & versions of the libraries used)
```

## Screenshots

### Home Page
![Home Page](./assets/Chatbot.png)
![Home Page2](./assets/Chatbot2.png)
![Home Page3](./assets/Chatbot3.png)

### Admin Pages
![Admin-Teachers](./assets/Admin-Teachers.png)
![Admin-Students](./assets/Admin-Students.png)
![Admin-Courses](./assets/Admin-Courses.png)
![Admin-Holidays](./assets/Admin-Holidays.png)