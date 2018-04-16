import nltk

data = {"This is a cat", "A cat is this.", "The cat sits."}
#text = data.split()
for sentence in data:
 #   print(sentence)
 #   for word in sentence:
 #       print(word)
 #   for i in range(len(sentence)):
     count = 0
     text = sentence.split()
     for word in text: 
      #  print(sentence[count])
        print(text[count -1], text[count])
        print(nltk.edit_distance(text[count-1], text[count]))
        count += 1
#print(nltk.edit_distance("Pickele", "Lickle"))



