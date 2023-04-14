# %%
import random
import nltk
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download('stopwords')

data_path = "./data/"
files = glob.glob(f"{data_path}*.txt")
print(f"The first 5 of {len(files)}", files[:5])


def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)


def load_doc(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text


def save_list(lines, filename):
    data = '\n'.join(lines)
    with open(filename, 'w') as file:
        file.write(data)


def random_sample(num1, num2):
    dataList = list(range(num1))
    TrainIndex = []
    for _ in range(num2):
        randIndex = int(random.uniform(0, len(dataList)))
        TrainIndex.append(dataList[randIndex])
        del (dataList[randIndex])
    TestIndex = dataList
    return TrainIndex, TestIndex


def load_doc_lines(filename):
    with open(filename, 'rt') as file:
        lines = []
        while 1:

            if line := file.readline():
                lines.append(line.strip("\n"))
            else:
                break
    return lines


def clean_doc(doc):
    tokens = doc.split()
    tokens = [word.lower() for word in tokens]
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def doc_to_line(doc):
    tokens = clean_doc(doc)
    return ' '.join(tokens)


def process_docs(files):
    lines = []
    for doc in files:
        line = doc_to_line(doc)
        lines.append(line)
    return lines


def prepare_data(train_docs, mode, vocab):
    vectorizer = CountVectorizer(vocabulary=vocab)
    transformer = TfidfTransformer(norm='l2')
    return transformer.fit_transform(vectorizer.fit_transform(train_docs))


# ArRe_train_lines = load_doc_lines(f"{data_path}/Reduced_ArtsReviews_5000.txt")
# %%
ArRe_train_lines = [load_doc_lines(file)[0] for file in files]
# %%
train_docs = process_docs(ArRe_train_lines)
# %%
vocab = []
for ll in train_docs:
    tt = ll.split()
    for ww in tt:
        if ww not in vocab:
            vocab.append(ww)

Xtrain = prepare_data(train_docs, 'tfidf', vocab)

trunc_SVD_model = TruncatedSVD(n_components=25)
approx_Xtrain = trunc_SVD_model.fit_transform(Xtrain)
print(f"Approximated Xtrain shape: {str(approx_Xtrain.shape)}")


def preprocess_query(review, mode, vocab):
    tokens = clean_doc(review)
    line = ' '.join(tokens)
    vectorizer = CountVectorizer(vocabulary=vocab)
    transformer = TfidfTransformer(norm='l2')
    return transformer.fit_transform(vectorizer.fit_transform([line]))


querys = ['The pen is good.', 'The pen is poor.']
Top_n_reviews = 10
for query in querys:
    encoded_query = preprocess_query(query, 'tfidf', vocab)
    transformed_query = trunc_SVD_model.transform(encoded_query)
    similarities = cosine_similarity(approx_Xtrain, transformed_query)
    indexes = np.argsort(similarities.flat)[::-1]

    print('\n' + 'Query: ' + query)
    for i in range(Top_n_reviews):
        print(f"Top {str(i + 1)} result:")
        print(f"Reviews ID: {str(indexes[i])}")
        print(ArRe_train_lines[indexes[i]])


# preprocess query
def preprocess_query(review, mode, vocab):
    # clean
    tokens = clean_doc(review)
    # convert to line
    line = ' '.join(tokens)
    # encode
    vectorizer = CountVectorizer(vocabulary=vocab)
    transformer = TfidfTransformer(norm='l2')
    return transformer.fit_transform(vectorizer.fit_transform([line]))


# Interplot Precision for standard Recall
def InterplotPrecision(p=0.1, Precision=None, Recall=None):

    if p >= 1.0:
        p = 0.9

    Mark = np.zeros(2)
    l = 0
    r = 0
    for i in range(len(Recall)):
        if Recall[i] >= p and Mark[0] == 0:
            l = i
            Mark[0] = 1
        if Recall[i] >= p + 0.1 and Mark[1] == 0:
            # if Recall[i] >= 1.0 and Mark[1] == 0:
            r = i
            Mark[1] = 1
    return max(Precision[l:(r + 1)])


# obtain y axis for R/P curve
def compute_RP_yaxis(Precision=None, Recall=None):
    y_axis = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(11):
        pInput = 0.1 * i
        y_axis[i] = InterplotPrecision(p=pInput,
                                       Precision=Precision,
                                       Recall=Recall)
    return y_axis


# compute Recall, Precision, F1-measure
def compute_R_P_F1(re_mark=None, QuRe_ID=None):
    Recall = []
    Precision = []
    F1measure = []
    for i in range(len(re_mark)):
        r = sum(re_mark[:(i + 1)])
        Re = r / (len(QuRe_ID))
        Pr = r / (i + 1)
        # avoid divisor to be 0
        FD = Re + Pr
        if FD == 0:
            FD = 1
        F1 = 2 * Re * Pr / FD

        Recall.append(Re)
        Precision.append(Pr)
        F1measure.append(F1)
    return Recall, Precision, F1measure


queries = ['The pen is good.', 'The pen is poor.']

re_ID = [[]]

AllRecall = []
AllPrecision = []
AllF1measure = []
for j, query in enumerate(queries):
    # retrieval
    encoded_query = preprocess_query(query, 'tfidf', vocab)
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
            ArRe_train_lines[indexes[i]],
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
    plt.plot(x_axis, y_axis, '-bo', color="purple", label="Query%d" % (j + 1))
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

# %%
