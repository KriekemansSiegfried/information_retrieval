import joblib
from nltk.corpus import stopwords
from include.bow import dictionary
from include.io import import_captions

from sklearn.feature_extraction.text import CountVectorizer

caption_filename = 'include/data/results_20130124.token'
visualize = False

# import captions from caption file (defined by filename above)
# create a dictionary (bow_dict), which is a set of the form
# {<word> : <#appearance_of_word> , <word> : <#appearance_of_word> , ...}
captions = import_captions.import_captions(caption_filename)
bow_dict = dictionary.create_dict(captions)
print('loaded {} captions'.format(len(captions)))

# if visualization turned on:
#   display the dictionary from least frequent to most frequent
#   display the dictionary from most frequent to least frequent
if visualize:
    df_word_freq = dictionary.rank_word_freq(dic=bow_dict, n=20, ascending=False, visualize=True)
    df_word_freq = dictionary.rank_word_freq(dic=bow_dict, n=20, ascending=True, visualize=True)
    df_word_freq = None

# get/import stop words
# prune the dictionary
stop_words = set(stopwords.words('english'))
a, b = dictionary.prune_dict(word_dict=bow_dict, stopwords=stop_words, min_word_len=3, min_freq=0, max_freq=1000)
bow_dict_pruned, removed_words = a, b

# create vectorizer
tokens = bow_dict_pruned.keys()
vectorizer = CountVectorizer(vocabulary=set(tokens))

# save vectorizer to .sav file
filename = 'include/data/vectorizer_model.sav'
joblib.dump(vectorizer, filename)


# testing
loaded_model = joblib.load(filename)
str_0 = "A girl is on rollerskates talking on her cellphone standing in a parking lot"
str_1 = "Two young guys with shaggy hair look at their hands while hanging out in the yard"
str_2 = "A child in a pink dress is climbing up a set of stairs in an entry way"
str_3 = "An asian man wearing a black suit stands near a dark-haired woman and a brown-haired woman"
str_4 = "A man with reflective safety clothes and ear protection drives a John Deere tractor on a road"
str_5 = "A young woman with dark hair and wearing glasses is putting white powder on a cake using a sifter"
str_6 = "A person in gray stands alone on a structure outdoors in the dark"
str_7 = "Man with a lit cigarette in mouth , yellow baseball cap turned backwards , and yellow shirt " \
        "over an orange polo shirt , helping another man with carrying slabs of concrete"
str_8 = "Three construction workers working on digging on a hole , while the supervisor looks at them"
str_9 = "Two workers in reflective vests walk in front of a wall painted with the image of a mannequin in a " \
        "reflective vest with rubber boots and a trowel"


test_strings = [str_0, str_1, str_2, str_3, str_4, str_5, str_6, str_7, str_8, str_9]
i = 0
for string in test_strings:
    v1 = vectorizer.transform([string])
    v2 = loaded_model.transform([string])
    if (v1.toarray() == v2.toarray()).all():
        print("test" + str(i) + " was successful")
    else:
        print("test" + str(i) + " failed")
    i = i+1







