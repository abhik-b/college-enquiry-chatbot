# Explanation of the basic concepts

My 1st plan is to create a neural network that learns through different patterns of text what user tries to mean through those texts (user intention). We are using neural net because the questions are **NOT hard-coded**. The users are free to type in the question in whichever form they wish to, as long it means the same. For example user1 says "What is my CGPA this sem ?" , chatbot understands user1 is asking for the result from the bot. User2 says "tell me my result", chatbot understands the user2 wants the result.

My next plan is to create a website where a chat interface is present. Now as the user types in his message our neural network gets the messge & predicts the user's intention & tells us. We then return response respective to the user intention. This website will give our system a _**look & feel**_ of a chatbot and our neural net will give the system _**intelligence & behaviour**_ like a chatbot.

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
 To be completed soon...

## Creating the Neural Network

Neural nets are a means of doing machine learning, in which a computer learns to perform some task by analyzing training examples. Usually, the examples have been hand-labeled in advance. An object recognition system, for instance, might be fed thousands of labeled images of cars, houses, coffee cups, and so on, and it would find visual patterns in the images that consistently correlate with particular labels.

We create some layers then we need a path to take data from each layer to another. A feedforward neural network (FNN) is an artificial neural network wherein connections between the nodes do not form a cycle. The feedforward neural network was the first and simplest type of artificial neural network devised.

I created a NeuralNet Class which actually derives the `torch.nn` module. It has 2 methods :
- `__init__`
: it creates the layers we need & also defines the activation function which is ReLU in our case. We are doing 4 layers. Each layer has a input & output size. First Layer will have length of X train's 1st bag of words as input size & hidden size as output size. The next 2 layers will have hidden sizes as input & output sizes. Last layer will have hidden size as input size but output size will be the no of tags we have.
- `forward`
: it helps in passing/forwarding the data with activation functions & outputting the data (without activation functions).
 
## Preparing The Dataset

First I open the json file & load all the contents from it. Then I get all the tags. Then for each pattern in each intent of intents array , I tokenize them & add them to all words array & xy array (alongside the tags they belong to).

Then I remove all the punctutations , stem every words , remove the duplicates & sort all words array. Then Remove the duplicates & sort the tags as well.

Then I create bag of words for each pattern in xy array & append it to X-Train dataset. Similarly I get index of the associated tag (for any pattern in xy array) in tags array & append it to Y-Train. Then convert both the trains to numpy arrays.

Next I created a ChatDataSet class deriving the Dataset Module from `torch.utils.data`. Here x_data & y_data is assigned & some minor utilites done.

The the dataset is loaded using dataloader,batch size is given 8 .Batch sizes are the data samples that will be used to train. For example I have 400 samples of data for training & I set batch_size = 8 then first 8 data samples (i.e from 1 to 8 ) will be used to train. Then again next 8 data samples (i.e from 9 to 16) will be used to train. Simliarly all 400 data samples will be trained by training 8 samples at once.Shuffle is set to true which means dataset will be shuffled.

Then we initialize the Neural Net Model & give it a device (cuda or cpu). We also need criterion & optimizer.
>`cuda`  is a parallel computing platform and application programming interface (API) that allows software to use certain types of graphics processing units (GPUs) for general purpose processing, an approach called general-purpose computing on GPUs (GPGPU).

## Training pipeline

1. Design the number of inputs , output size & forward pass.
2. Construct the loss & optimizer
3. Training loop
	- forward pass : passing data forward in layers (compute prediction) 
	- backward (gradient)
	- update weights & bias


### Training Loop 
X data = words, Y data = labels

For each epoch I convert the words & labels to CUDA device then again convert labels to `long` data type(dtype). Now it is ready for training.

1. `model(words)` : this calls the forward method , thus passing the x data in feed forward network. When you call the model directly this Forward is called , the internal  `__call__`  function is used. This function manages all registered hooks and calls forward afterwards. That’s also the reason you should call the model directly, because otherwise your hooks might not work etc.
2. Then calculate loss using `CrossEntropyLoss` . A **loss function** operates on the error to quantify how bad it is to get an error of a particular size/direction, which is affected by the negative consequences that result in an incorrect prediction. [Read here for more info !](https://www.google.com/amp/s/neptune.ai/blog/cross-entropy-loss-and-its-applications-in-deep-learning/amp)
3. Then optimize the weights. *(See Below the optimization section)*

### Some points to note :

- **one epoch** = one forward pass and one backward pass of all the training examples
- **batch size** = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
- **number of iterations** = number of passes, each pass using batch size as number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).

	**Example**: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

### Optimization

An Optimizer is used to update weights. An optimization algorithm finds the value of the parameters(weights) that minimize the error when mapping inputs to outputs. These optimization algorithms or optimizers widely affect the accuracy of the deep learning model.

**Adam**, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters.

**Learning Rate** : we use `0.001` as learning rate. Learning Rate helps to opimize to lower the loss but only take certain size steps to opimize (not gigantic steps because then the optimization might stuck). 
![](https://abhik-b.github.io/exam-notes/assets/images/2022-09-21-14-43-36.png)

All optimizers implement a step() method, that updates the parameters.
`optimizer.step()` 

- `zero_grad` clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
- `loss.backward()` computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
- `optimizer.step()` causes the optimizer to take a step based on the gradients of the parameters.

## After Training
I create a data object which stores :
- Model state
- input , output, hidden size
- All words array
- Tags array
Then save this data object to a pth file. A common PyTorch convention is to save models using either a `.pt` or `.pth` file extension.

> A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor. [Read more](https://pytorch.org/tutorials/beginner/saving_loading_models.html) 


## Predict Intention

You must call `model.eval()` to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.

Then created a function `chatbot_response` which will predict the tag of the message passed to the function as argument. 

The input message is first tokenized , then a bag of words is created for that sentence. Then I reshape the array with 1 row (because we only havbe 1 sample) & same number of columns as bag of words row. Then I convert this numpy array to tensor.


Then I get the predictions from model by passing X. Then I get the the maximum value of all elements in the X tensor.Not `torch.max` returns 2 values , one is values & other is indices. We are interested in **indices tensor**  because the index it returns is the respective index of the tag chatbot predicts for the input message.Then I get the tag from tags array by passing the index I recieved from indices array. 

Finally I manually apply softmax to the prediction. The softmax transforms values like *positive, negative, zero, or greater than one* into values between 0 and 1, so that they can be interpreted as **probabilities**. So it returns a 2D tensor whose columns have the probabilites for each tag. Thus we get the probability for the predicted tag. We will return response only if the probabilty is greater than 75% otherwise not.


## Interacting with Chatbot

In Frontend, as soon as user types in the message & hits the send button , a POST request is made to the chatbot api. Then the response is sent back to the user. Then a div is created which will have the message.  If the response has :
- ***data object*** i.e it is a link to a file , then  another div is added which when clicked will open the link.
- ***url*** then the next user message will be sent to this url 

In Backend , there are 2 API Endpoints. One is for normal chatting & other one is to return results for a particular id. So when user sends a message to chatbot , it goes to the `chatbot_response` & user intention (tag) is predicted. Then according to the tag the response is given. For some tags we need to send a pdf file , so we prepare a link & send it alongside the response. For result we send back the second api route as url. For all other tags we send the response. For some tags we already have some response in `intents.json`.
