## How will we do it

My 1st plan is to create a neural network that learns through different patterns of text what user tries to mean through those texts (user intention). We are using neural net because the questions are **NOT hard-coded**. The users are free to type in the question in whichever form they wish to, as long it means the same. For example user1 says "What is my CGPA this sem ?" , chatbot understands user1 is asking for the result from the bot. User2 says "tell me my result", chatbot understands the user2 wants the result.

My next plan is to create a website where a chat interface is present. Now as the user types in his message our neural network gets the messge & predicts the user's intention & tells us. We then return response respective to the user intention. This website will give our system a _look & feel_ of a chatbot and our neural net will give the system _intelligence & behaviour_ like a chatbot.

So we will do the following :

1. Create a dataset for training the neural net
2. Basic NLP
3. Basic feed forward neural network
4. Predict Intention
5. Flask REST api server
6. html css js frontend for chat
7. database models to store info on teachers,students & courses
8. html css admin dashboard

To do the above , we need :

- database (sqlite3),
- flask rest api server ,
- html css js frontend ,
- neural network model ,
- training the model,
- dataset preparation,
- chat logic


## Creating Datasets (JSON)

A dataset is what we are going to use to train the neural network model. Like any other dataset this will also have labels(y) & features(X).
![intents.json](/screenshots/Screenshot%202022-09-11%20120841.png)

As you can see above our `intents.json` has a `intents` array. Each object or intent in this array has :

1. tag - this is what user is saying for the following input patterns.
2. patterns - these are patterns of user input which will be used to predict the above tag.
3. responses - these are the following responses to the tag

## Basic Natural Language Processing
 Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.
 
 Here we will derive meaningful information from natural language text. It will involve in structuring the input text (tokenization & stemming) , deriving patterns within the structured data (bag of words) & finally evaluating & interpreting the output.

 While training the dataset , First we will break each sentence into words then stem them. Then prepare a array of all these stemmed words. Then train the dataset.
 
 Later when user inputs a sentence then we will tokenize that statement also & prepare a bag of words array with respect to the all words array & this help in predicting the user intention. 
  
 ### Tokenization

  Tokenization is a way of separating a piece of text into smaller units called tokens. Here, tokens can be either words, characters, or sentences. For example, consider the sentence: “Never give up”.Assuming space as a delimiter, the tokenization of the sentence results in 3 tokens – Never-give-up. As each token is a word, it becomes an example of Word tokenization.

  In this project, we will use `nltk.word_tokenize(sentence)` to tokenize each sentence into words.

 ### Stemming
 Stemming is used as an approximate method for grouping words with a similar basic meaning together. For example, a text mentioning "daffodils" is probably closely related to a text mentioning "daffodil" (without the s).
 
 **Example** : 
 - words : "going","goes","gone" 
 - stem : "go"

 So in the above example , all those 3 words have the same stem or loosely same meaning. So stemming the words makes sense in this case.

 We are using **Porter Stemmer** algorithms. This stemmer algorithms are known for its speed and simplicity.

 ### Bag of words
 The bag-of-words model is a simplifying representation used in natural language processing and information retrieval (IR). In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity.

 [Watch This Video for more details](https://youtu.be/IKgBLTeQQL8)

 The bag of words array will have 1 for each known word (word in all words array) that exists in the given tokenized sentence & 0 for other words.

Example:
```python
all_words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
sentence="hello how are you"
tokenized_sentence = ["hello", "how", "are", "you"]
bag   = [  0 ,  1 , 0 , 1 , 0 ,  0 ,  0]
# This means that hello & you are present in bag because
# 1 is present only in those index of hello & you,
# all the rest are 0 meaning the rest of the 
# words in all words array are not present in the sentence.
```



 ### Pattern Matcher (Optional)


## Creating the Neural Network

Neural nets are a means of doing machine learning, in which a computer learns to perform some task by analyzing training examples. Usually, the examples have been hand-labeled in advance. An object recognition system, for instance, might be fed thousands of labeled images of cars, houses, coffee cups, and so on, and it would find visual patterns in the images that consistently correlate with particular labels.

We create some layers then we need a path to take data from each layer to another. A feedforward neural network (FNN) is an artificial neural network wherein connections between the nodes do not form a cycle. The feedforward neural network was the first and simplest type of artificial neural network devised.

I created a NeuralNet Class which actually derives the `torch.nn` module. It has 2 methods :
- `__init__`
: it creates the layers we need & also defines the activation function which is ReLU in our case. We are doing 4 layers. Each layer has a input & output size. First Layer will have length of X train's 1st element as input size & hidden size as output size. The next 2 layers will have hidden sizes as input & output sizes. Last layer will have hidden size as input size but output size will be the no of tags we have.
- `forward`
: it helps in passing/forwarding the data with activation functions & outputting the data (without activation functions).
 
## Preparing The Dataset

First I open the json file & load all the contents from it. Then I get all the tags. Then for each pattern in each intent of intents array , I tokenize them & add them to all words array & xy array (alongside the tags they belong to).

Then I remove all the punctutations , stem every words , remove the duplicates & sort all words array. Then Remove the duplicates & sort the tags as well.

Then I create bag of words for each pattern in xy array & append it to X-Train dataset. Similarly I get index of the associated tag (for any pattern in xy array) in tags array & append it to Y-Train. Then convert both the trains to numpy arrays.

Next I created a ChatDataSet class derving the Dataset Module from `torch.utils.data`. Here x_data & y_data is assigned 


## Predict Intention

## Create REST api

