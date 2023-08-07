import pandas as pd 
import cv2
import nltk
nltk.download('averaged_perceptron_tagger')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textstat import flesch_reading_ease
from tabulate import tabulate
import numpy as np

data = pd.read_csv('image_data.csv') # reads the image data file


# Iterate over each row in the CSV file

prompt_lengths = []

for index, row in data.iterrows():
    #Load the image using the file path
    # image = cv2.imread(row['file'])
    # cv2.imshow("Image", image) # this line shows the image
    # cv2.waitKey(2000) # waits before closing the image
    

    words = nltk.word_tokenize(row['prompt'])
    num_words = 0
    for i in words:
        if i == '.':
            continue
        else:
            num_words += 1

    prompt_lengths.append(num_words) # creates a list of prompt lengths
    
    #cv2.destroyAllWindows()






data["prompt_length(words)"] = prompt_lengths # creates a new column in the dataframe called "prompt_lengths(words)"
data = data.rename(columns={"file": "image"})
print(data.head())


# Finding the maximum length prompt
largest_prompt = data["prompt_length(words)"].max() # finds the maximum value of the column
 # print(largest_prompt)

# Creating a Word Cloud
text_data = data['prompt'].str.cat(sep=' ')
wordcloud = WordCloud(width=800, height=400).generate(text_data)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# Creating a 'readability' column into the dataframe

data['readability'] = data['prompt'].apply(lambda x: flesch_reading_ease(x))


# Plotting the graph between length and readbility
plt.scatter(data['prompt_length(words)'], data['readability'])
plt.xlabel('Length (in words)')
plt.ylabel('Readability')
plt.title('Length vs. Readability')
plt.grid(True)

coefficients = np.polyfit(data['prompt_length(words)'], data['readability'], 1)
line = np.polyval(coefficients, data['prompt_length(words)'])
plt.plot(data['prompt_length(words)'], line, color='red')


plt.show()

# Sentence Complexity Function

def calculate_sentence_structure_complexity(prompt):
    # Tokenize the prompt into sentences
    sentences = nltk.sent_tokenize(prompt)
    
    # Initialize a list to store the complexity scores
    complexity_scores = []
    
    # Iterate over each sentence
    for sentence in sentences:
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        
        # Perform part-of-speech tagging
        tagged_words = nltk.pos_tag(words)
        
        # Calculate the complexity score based on the number of noun phrases
        noun_phrases = [word for word, tag in tagged_words if tag.startswith('NN')]
        complexity_scores.append(len(noun_phrases))
    
    return complexity_scores

# Adding the Sentence Complexity Column to the Dataframe
data['complexity_scores'] = data['prompt'].apply(calculate_sentence_structure_complexity)
print(data.head())