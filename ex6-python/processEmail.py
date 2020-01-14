import re
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from getVocabList import getVocabList 

def processEmail(email_contents):
    vocabList = getVocabList()
    word_indices = []

    # ========================== Preprocess Email ===========================

    # Find the Headers ( \n\n and remove )
    # Uncomment the following lines if you are working with raw emails with the
    # full headers

    # hdrstart = email_contents.find("\n\n")
    # if hdrstart:
    #     email_contents = email_contents[hdrstart:]
    
    # Lower case
    email_contents = email_contents.lower()

    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)

    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)

    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)

    # Output the email to screen as well
    print('\n==== Processed Email ====\n\n')
    # Process file
    l = 0

    # Split and also get rid of any punctuation
    # regex may need further debugging...
    email_contents = re.split(r'[@$/#.-:&\*\+=\[\]?!(){},\'\'\">_<;%\s\n\r\t]+', email_contents)
    for token in email_contents:
        # Remove any non alphanumeric characters
        token = re.sub('[^a-zA-Z0-9]', '', token)
        # Stem the word 
        token = PorterStemmer().stem(token.strip())
        # Skip the word if it is too short
        if len(token) < 1:
           continue
        idx = vocabList[token] if token in vocabList else 0
        # only add entries which are in vocabList
        #   i.e. those with ind ~= 0, 
        #        given that ind is assigned 0 if str is not found in vocabList
        if idx > 0:
            word_indices.append(idx)
        # Print to screen, ensuring that the output lines are not too long
        if l + len(token) + 1 > 78:
            print("")
            l = 0

        print('{:s}'.format(token)),
        l = l + len(token) + 1

    # Print footer
    print('\n\n=========================\n')
    return word_indices