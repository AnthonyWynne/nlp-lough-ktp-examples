import random
import nltk
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import os
import shutil
import argparse

def flatten_dirs(path:str):
    source_folder = path
    destination_folder = path

    # Iterate through all subfolders and files in the source folder
    for root, files in os.walk(source_folder):
        for file in files:
            # Get the absolute path of the file
            file_path = os.path.join(root, file)
            
            # Move the file to the destination folder
            shutil.move(file_path, destination_folder)
            print(f"Moved file: {file_path}")

def flatten_folder_tree(source_dir, destination_dir):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_dir, file)

            shutil.move(source_file, destination_file)
            print(f"Moved {source_file} to {destination_file}")

        for subdir in dirs:
            source_subdir = os.path.join(root, subdir)
            destination_subdir = os.path.join(destination_dir, subdir)

            os.makedirs(destination_subdir, exist_ok=True)
            print(f"Created directory: {destination_subdir}")


def load_doc_lines(filename):
    """Load all lines in a file into memory

    Args:
        filename (str): path to the file

    Returns:
        list: list of lines in the file

    """
    with open(filename, 'rt') as file:
        lines = []
        while 1:
            if line := file.readline():
                lines.append(line.strip("\n"))
            else:
                break
    return lines

def process_docs(files):
    """Process all docs in a directory

    Args:
        files (list): list of paths to files

    Returns:
        list: list of cleaned docs
    """
    lines = []
    for doc in files:
        line = doc_to_tokens(doc)
        lines.append(line)
    return lines

def doc_to_tokens(doc):
    """Clean a document string using the
    a series of functions to split, stem,
    remove punctuation and stopwords.

    Args:
        doc (str): the doc to clean

    Returns:
        list: list of cleaned tokens
    """
    tokens = clean_doc(doc)
    return ' '.join(tokens)

def clean_doc(doc):
    """Clean a document string using the
    a series of functions to split, stem,
    remove punctuation and stopwords.

    Args:
        doc (str): the doc to clean

    Returns:
        list: list of cleaned tokens
    """
    tokens = Split_doc(doc)
    tokens = porter_stem(tokens)
    tokens = make_trans_table(tokens)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = remove_stopwords(tokens)
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def Split_doc(doc):
    """Split a long document string into
    basic tokens splitting on whitespace.

    Args:
        doc (str): the doc to split

    Returns:
        list: list of tokens
    """
    tokens = doc.split()
    tokens = [word.lower() for word in tokens]
    return tokens

def porter_stem(tokens):
    """Stem tokens using the Porter Stemming algorithm

    Args:
        tokens (list): list of tokens

    Returns:
        list: list of stemmed tokens
    """
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    return tokens

def make_trans_table(tokens):
    """Create a translation table and remove punctuation
    from a list of tokens

    Args:
        tokens (list): list of tokens

    Returns:
        list: list of tokens without punctuation
    """
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    return tokens

def remove_stopwords(tokens):
    """Remove stopwords from a list of tokens
    Args:
        tokens (list): list of tokens

    Returns:
    list: list of tokens without stopwords
    """
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

def prepare_data(train_docs, vocab):
    """Prepare the data for training

    Args:
        train_docs (list): list of training docs
        vocab (set): the vocabulary to use

    Returns:
        csr_matrix: the preprocessed training data
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    transformer = TfidfTransformer(norm='l2')
    return transformer.fit_transform(vectorizer.fit_transform(train_docs))

def preprocess_query(review, vocab):
    """Preprocess a query

    Args:
        review (str): the query to preprocess
        mode (str): the mode to use
        vocab (set): the vocabulary to use

    Returns:
        csr_matrix: the preprocessed query
    """
    tokens = clean_doc(review)
    line = ' '.join(tokens)
    vectorizer = CountVectorizer(vocabulary=vocab)
    transformer = TfidfTransformer(norm='l2')
    return transformer.fit_transform(vectorizer.fit_transform([line]))

def run_lsi(data_path:str,queries:[str],Top_n_reviews:int):

    nltk.download('stopwords')

    files = glob.glob(f"{data_path}*")

    train_lines = [load_doc_lines(file)[0] for file in files]
    train_docs = process_docs(train_lines)    
    
    vocab = []
    for ll in train_docs:
        tt = ll.split()
        for ww in tt:
            if ww not in vocab:
                vocab.append(ww)

    Xtrain = prepare_data(train_docs, vocab)

    trunc_SVD_model = TruncatedSVD(n_components=25)
    approx_Xtrain = trunc_SVD_model.fit_transform(Xtrain)
    print(f"Approximated Xtrain shape: {str(approx_Xtrain.shape)}")

    for query in queries:
        encoded_query = preprocess_query(query, vocab)
        transformed_query = trunc_SVD_model.transform(encoded_query)
        similarities = cosine_similarity(approx_Xtrain, transformed_query)
        indexes = np.argsort(similarities.flat)[::-1]

        print('\n' + 'Query: ' + query)
        for i in range(Top_n_reviews):
            print(f"Top {str(i + 1)} result:")
            print(f"Reviews ID: {str(indexes[i])}")
            print(train_lines[indexes[i]])
    
    return queries,vocab

def model_analysis(queries,vocab):
    re_ID = [[]]

    AllRecall = []
    AllPrecision = []
    AllF1measure = []
    for j, query in enumerate(queries):
        # retrieval
        encoded_query = preprocess_query(query, vocab)
        transformed_query = trunc_SVD_model.transform(encoded_query)
        similarities = cosine_similarity(approx_Xtrain, transformed_query)

        # rank the index
        indexes = np.argsort(similarities.flat)[::-1]

        # Mark the relevant index
        re_mark = []
        for i in range(len(indexes)):
            if (indexes[i] + 1) in re_ID[j]:
                re_mark.append(1)
            else:
                re_mark.append(0)
        # print(re_mark)

        # compute Recall, Precision, F1-measure
        Recall, Precision, F1measure = compute_R_P_F1(re_mark=re_mark,
                                                      QuRe_ID=re_ID[j])

        print('\n' + 'Query%d: ' % (j + 1) + query)
        for i in range(10):
            print(
                f"Top {str(i + 1)}" + ' result: ID%d ' % (indexes[i] + 1),
                train_lines[indexes[i]],
            )
        Recall = np.array(Recall)
        Precision = np.array(Precision)
        F1measure = np.array(F1measure)
        # print(re_mark)
        print("Recall@1~10: ", np.around(Recall[:10], 2))
        print("Precision@1~10: ", np.around(Precision[:10], 2))
        print("F1measure@1~10: ", np.around(F1measure[:10], 2))

        # save
        AllRecall.append(Recall)
        AllPrecision.append(Precision)
        AllF1measure.append(F1measure)

        # plot R/P curve
        x_axis = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y_axis = compute_RP_yaxis(Precision=Precision, Recall=Recall)
        plt.plot(x_axis,
                 y_axis,
                 '-bo',
                 color="purple",
                 label="Query%d" % (j + 1))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Standard Recall/Precision Curves')
        plt.legend()
        plt.show()

    # compute average Recall, average Precision, average F1-measure
    AllRecall = np.array(AllRecall)
    AllPrecision = np.array(AllPrecision)
    AllF1measure = np.array(AllF1measure)
    AveRecall = (AllRecall[0] + AllRecall[1]) / 2
    AvePrecision = (AllPrecision[0] + AllPrecision[1]) / 2
    AveF1measure = (AllF1measure[0] + AllF1measure[1]) / 2

    print("\nAverage Recall, average Precision, average F1-measure: ")
    print("average Recall@1~10: ", np.around(AveRecall[:10], 2))
    print("average Precision@1~10: ", np.around(AvePrecision[:10], 2))
    print("average F1measure@1~10: ", np.around(AveF1measure[:10], 2))

    # plot average R/P curve
    x_axis = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y_axis = compute_RP_yaxis(Precision=AvePrecision, Recall=AveRecall)
    plt.plot(x_axis, y_axis, '-bo', color="blue", label="Average")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('average Recall')
    plt.ylabel('average Precision')
    plt.title('Standard Average Recall/Precision Curves')
    plt.legend()
    plt.show()

def main():
    #take arguments in line, replace the hard-coded vars in source
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', help='target directory',default='data/')
    parser.add_argument('-q','--queries', help='comma seperated list of string phrases to use as queries.')
    parser.add_argument('-n','--Top_n_reviews', help='number of results', default=10)
    parser.add_argument('-a','--analysis', help='boolean flag, true to check model efficacy',default='False')
    args = parser.parse_args()
    
    data_path = args.data_path
    queries = args.queries.split(',')
    Top_n_reviews = int(args.Top_n_reviews)
    analysis = args.analysis
    queries,vocab = run_lsi(data_path,queries,Top_n_reviews)

    if analysis.lower =='true':
        model_analysis(queries,vocab)


    


if __name__ == '__main__':
    main()