from transformers import AutoTokenizer,AutoModelForSequenceClassification
from scipy.special import softmax

tweet = "wow! such a nice day @kaartikay. check it at https://youtube.com"

words_in_tweet = []

for word in tweet.split(" "):
    
    if word.startswith('http'):
        word = "http"
    elif word.startswith("@") and len(word)>1 :
        word = "@user"
    words_in_tweet.append(word)
tweet_in_sen = " ".join(words_in_tweet)

# the model
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)

# now we load the tokenizer so we can convert tweet text to appropriate number values
tokenizer = AutoTokenizer.from_pretrained(roberta)

# now to get the lables of the outputs of mymodel
labels = ['Negative', 'Neutral', 'Positive']

# sentiment analysis where we first convert the processed tweet in pytorch tenson then pass that into the model
encoded_tweet = tokenizer(tweet_in_sen, return_tensors='pt')
print(encoded_tweet)
# output for printing the encoded tweet is a dictionary that has tensors which comes from coverting tweets into numbers 
# = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])

output = model(**encoded_tweet)

scores = output[0][0].detach().numpy()
scores = softmax(scores)

for i in range(len(scores)):
    
    l = labels[i]
    s = scores[i]
    print(l,s)