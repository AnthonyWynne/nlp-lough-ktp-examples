# %%
"""Generate text files from the articles in a sqlite database
"""
import sqlalchemy
import re

# Create engine: engine
engine = sqlalchemy.create_engine('sqlite:///./data/text.db')

# query to get the articles ids
query = "SELECT pageId FROM articles"


def clean_text(text):
    """strip punctuation, spaces and non printing
    character from the text to format for a filename

    Args:
        text (string): the input text

    Returns:
        string: lowercase text with no spaces or punctuation
    """
    text = re.sub(r'[^\w\s]', '', text).lower()
    text = re.sub(r'\s+', '_', text)
    return text.strip()


# get the articles ids
articles_ids = engine.execute(query).fetchall()

# query to get the articles
articles_limit = 100
for i, article_id in enumerate(articles_ids):
    query = f"SELECT text FROM articles WHERE pageId = {article_id[0]}"
    article = engine.execute(query).fetchone()
    # get the article title
    query = f"SELECT title FROM articles WHERE pageId = {article_id[0]}"
    title = engine.execute(query).fetchone()
    # clean the title
    filename = clean_text(title[0])
    # write the article to a file
    with open(f"./data/{filename}.txt", "w") as file:
        file.write(article[0])
    if i == articles_limit:
        break

# %%
