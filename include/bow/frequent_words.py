import pandas as pd
import matplotlib.pyplot as plt


def rank_word_freq(dic, n=15, ascending=False, visualize=True):
    
    """
    constructs a pandas dataframe with the the number of words in descending (ascending)
    order
    

    Parameters
    ----------
    dic : dictionary
        dictonary containing the words and their frequencys
        
    n: integer
        number of words to show on the figure 
        (default is 15)
        
    ascending : boolean 
        whether dataframe with their word counts should be shown in 
        ascending or descending order
        (default is False)
        
    visualize: boolean 
        indicating to make a figure showing the word count
        (default is True)


    Returns
    -------
    pandas dataframe with the top n words sorted in decending (ascending) 
    order according to their word count

    """
    
    # convert dictionary to pandas (pd) dataframe and
    # sort values based on counts
    df = (pd.DataFrame.from_dict(dic, 
          orient='index', columns=['count']).
          reset_index().rename(columns={'index':'word'}).
          sort_values(by=['count'],ascending=ascending))
                         
                         
    if visualize:
        df.iloc[0:n,:].plot.barh(x='word', y='count', legend=None)
        plt.xlabel('Word counts')
        plt.ylabel('Word')



    
    return df