import random
import json
import torch
from neural_net import NeuralNet
from nlp_utils import bag_of_words,tokenize

with open('intents.json','r') as json_data:
    intents = json.load(json_data)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE="data.pth"
data=torch.load(FILE)

input_size=data['input_size']
output_size=data['output_size']
hidden_size=data['hidden_size']
all_words=data['all_words']
tags=data['tags']
model_state=data['model_state']

model = NeuralNet(input_size, hidden_size,output_size)
model.load_state_dict(model_state)
model.eval()
model.to(device)

def chatbot_response(sentence):
        tokenized_sentence=tokenize(sentence)
        X = bag_of_words(tokenized_sentence, all_words)
        X = X.reshape(1,X.shape[0])
        X = torch.from_numpy(X)

        X = X.to(device)

        output=model(X)
        _, predicted=torch.max(output,dim=1)
        tag=tags[predicted.item()]
        
        probs=torch.softmax(output,dim=1)
        prob=probs[0][predicted.item()]

        if prob.item() > 0.75 :
            for intent in intents['intents'] :
                if tag == intent['tag']:
                    return (random.choice(intent['responses']),tag)
        return ("I am unable to understand that..",'greetings')

print('hi')