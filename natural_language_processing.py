import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk import ne_chunk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Sample text for analysis
text = "Natural Language Processing is a subfield of artificial intelligence that focuses on the interaction between computers and humans using natural language."

# Tokenization
tokens = word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

# Part-of-Speech Tagging
pos_tags = pos_tag(filtered_tokens)

# Named Entity Recognition
named_entities = ne_chunk(pos_tags)

# Frequency Distribution
freq_dist = FreqDist(filtered_tokens)

# Display results
print("Original Text:")
print(text)

print("\nTokenization:")
print(tokens)

print("\nStopword Removal:")
print(filtered_tokens)

print("\nStemming:")
print(stemmed_tokens)

print("\nPart-of-Speech Tagging:")
print(pos_tags)

print("\nNamed Entity Recognition:")
print(named_entities)

print("\nWord Frequency Distribution:")
print(freq_dist.most_common())
