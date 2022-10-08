# How will we do it

My 1st plan is to create a neural network that learns through different patterns of text what user tries to mean through those texts. We are using neural net because the questions are **NOT hard-coded**. The users are free to type in the question in whichever form they wish to, as long it means the same. For example user1 says "What is my CGPA this sem ?" , chatbot understands user1 is asking for the result from the bot. User2 says "tell me my result", chatbot understands the user2 wants the result.

My next plan is to create a website where a chat interface is present. Now as the user types in his message our neural network gets the messge & predicts the user's intention & tells us. We then return response respective to the user intention. This website will give our system a look & feel of a chatbot and our neural net will give the system intelligence & behaviour like a chatbot.

So we will do the following :

1. create a dataset for training the neural net
2. basic NLP
3. basic feed forward neural network
4. implementing chat
5. flask rest api server
6. html css chat frontend for chat
7. database models to store info on teachers,sdtudents & courses
8. html css admin dashboard

To do the above , we need :

- database (sqlite3),
- flask rest api server ,
- html css js frontend ,
- neural network model ,
- training the model,
- dataset preparation,
- chat logic


## Getting started

First install `virtualenv` then create a virtual environment using `virtual venv` command in **terminal/cmd prompt/powershell**. Then activate it using `venv\Scripts\activate`. Now we can install our dependencies for this project inside this isolated virtual environment.

We will install the following libraries to get started :

- **torch** - to create the neural network
- **nltk & spacy** - for NLP (natural language processing)
- **flask** - to run the web server where
  - we will interact with our chatbot
  - admin dashboard to manage our teachers, students, courses

So I went to Pytorch's [ official guide ](https://pytorch.org/get-started/locally/)to start local developement & got the cmd I need to install pytorch with cuda.

I used `pip install nltk spacy flask` to install them together. Then I use `python -m spacy download en_core_web_sm` to download english core small version.

Then I saved the names & versions of everything I use in a _requirements.txt_ file by using `pip freeze > requirements.txt`

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

## Creating Datasets

A dataset is what we are going to use to train the neural network model. Like any other dataset this will also have labels(y) & features(X).
![intents.json](/screenshots/Screenshot%202022-09-11%20120841.png)

As you can see above our `intents.json` has a `intents` array. Each object or intent in this array has :

1. tag - this is what user is saying for the following input patterns.
2. patterns - these are patterns of user input which will be used to predict the above tag.
3. responses - these are the following responses to the tag

## Basic Natural Language Processing

## Creating the Neural Network