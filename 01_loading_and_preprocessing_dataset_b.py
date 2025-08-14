import glob
import os
import pandas as pd
from striprtf.striprtf import rtf_to_text

# Set directory
os.chdir('/home/t/Documents/uni/Master/SoSe25/FP - GenAI') 


def read_rtf_file(file_path):
    with open(file_path, 'r') as file:
        rtf_text = file.read()
    plain_text = rtf_to_text(rtf_text)
    return plain_text


# Generate a list of all .rtf files in the specified directory (exclude doclist file)
file_name = [rtf for rtf in glob.glob('Hausarbeit2.0/Data/Artikel_FR_Welt_B/*.RTF') if not os.path.basename(rtf).startswith('Dateien (')]

# Sort the list of file names
file_name = sorted(file_name)


# Read all the articles and save their content to a list
articles_list = []
for name in file_name:
    article = read_rtf_file(name)
    articles_list.append(article)
    
    
# Create a dataframe to store the articles and their file names
articles_df = pd.DataFrame()
articles_df["text"] = articles_list


# Extract the date and clean text from each article
clean_text = [] #main article content
metadata = [] #metadata
body_text = []

for article in articles_df["text"]:
    part1, sep, part2 = article.partition('Body')
    clean_text.append(part2.strip() if sep else article.strip())
    metadata.append(part1.strip() if sep else None)
    
articles_df["metadata"] = metadata
articles_df["body_text"] = clean_text


# Add title
articles_df["title"] = articles_df["metadata"].str.extract(r'^(.*?)\s+(?=Die Welt|Frankfurter Rundschau)')

# Extract date and save it to the date column
date = []
for article in articles_df["text"]:
    if isinstance(article, str) and "Load-Date:" in article:
        extracted = (
            article.partition("Load-Date:")[2]
            .replace("End of Document", "")
            .replace("Ende des Dokuments", "")
            .replace("Original Gesamtseiten-PDF", "")
            .strip()
        )
    else:
        extracted = None
    date.append(extracted)
    
articles_df["date"] = date

# Convert to datetime
articles_df['date'] = pd.to_datetime(articles_df['date'], format='%B %d, %Y', errors='coerce')

# Define the cutoff date
cutoff = pd.Timestamp('2023-06-01')

# Filter for articles on or after 1 June 2023
articles_df = articles_df[articles_df['date'] >= cutoff]

# Reset index
articles_df.reset_index(drop=True, inplace=True)

# Drop rows where date conversion failed
articles_df = articles_df.dropna(subset=['date'])

# Create a 'month' column (year + month)
articles_df['month'] = articles_df['date'].dt.to_period('M')


# Remove unnecessary information from the texts
import re

def clean_newspaper_text(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'(?i)Link zum PDF', '', text)  # rm unnecessary information
    text = re.sub(r'(?i)Alle Rechte vorbehalten', '', text)
    text = re.sub(r'(?i)Ende des Dokuments', '', text) 
    text = re.sub(r'(?i)Original Gesamtseiten-PDF', '', text)
    text = re.sub(r'Copyright.*?\n', '', text)  # rm copyright
    text = re.sub(r'(Section:.*?\n|Length:.*?\n|Load-Date:.*?\n)', '', text)  # rm load-date
    text = re.sub(r'\n{3,}', '\n\n', text)  # rm triple linebreaks
    text = re.sub(r'[ ]{2,}', ' ', text)  # rm multiple spaces
    text = re.sub(r'\n\s+', ' ', text)  # turn line breaks + spaces into a single space
    text = text.replace('\xad', '')  # rm formating errors
    text = text.replace('\xa0', ' ')  # rm non-breaking spaces
    return text.strip()

articles_df["body_text"] = articles_df["body_text"].apply(clean_newspaper_text)


# Add newspaper column
import numpy as np
articles_df["newspaper"] = np.select(
    [
        articles_df["metadata"].str.contains("Die Welt", case=False, na=False),
        articles_df["metadata"].str.contains("Frankfurter Rundschau", case=False, na=False)
    ],
    [
        "Welt",
        "FR"
    ],
    default=None  # use None here, not np.nan
).astype(object)

# Drop rows where newspaper is missing
articles_df = articles_df.dropna(subset=["newspaper"])


#Remove duplicates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Extract text column
texts = articles_df['body_text'].astype(str).tolist()

# Step 2: Vectorize with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2)
tfidf_matrix = vectorizer.fit_transform(texts)

# Step 3: Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix)

# Step 4: Identify duplicates (90% or more similarity)
to_drop = set()
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        if cosine_sim[i, j] >= 0.90:
            to_drop.add(j)

# Step 5: Drop duplicates from the DataFrame
articles_df_deduplicated = articles_df.drop(articles_df.index[list(to_drop)]).reset_index(drop=True)

# Add article id
articles_df_deduplicated["article_id"] = articles_df_deduplicated.index

# Add group
articles_df_deduplicated['group'] = 'B'


# Separate articles by paragraph

# Regex pattern dataset B
pattern = re.compile(
    r'(?i)'
    r'(?=.*\b(?:Fachkräft\w*|Fachkräf\w*|Arbeitskräfte?(?:mangel|engpass|not|problem|knappheit)?|Personal(?:mangel|engpass|not|problem|knappheit)?|qualifizierte\w*\s+Arbeitskräfte?)\b)'
    r'(?=.*\b(?:Migrat\w*|Migrant\w*|Zuwanderung\w*|Einwanderung\w*|Geflüchtet\w*|Flüchtling\w*|Drittstaat\w*)\b)'
)

filtered_rows = []

for idx, row in articles_df_deduplicated.iterrows():
    text = row["body_text"]
    if not isinstance(text, str):
        continue

    total_questions = text.count('?') # Count ? to identify interviews (defined as documents with more than 10 question marks)
    paragraphs = re.split(r'\r?\n(?=[A-ZÄÖÜ]| )', text)

    # Decide whether to merge Q&A paragraphs
    do_merge = any("interview" in p.lower() for p in paragraphs) or (total_questions > 10)

    if do_merge:
        merged_paragraphs = []
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i].strip()
            if para.endswith("?"):
                merged_para = para
                i += 1
                # Merge all following paragraphs until next question or end
                while i < len(paragraphs) and not paragraphs[i].strip().endswith("?"):
                    merged_para += " " + paragraphs[i].strip()
                    i += 1
                merged_paragraphs.append(merged_para)
            else:
                merged_paragraphs.append(para)
                i += 1
        paragraphs = merged_paragraphs
        

    # Filter paragraphs by regex pattern
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if pattern.search(paragraph):
            new_row = row.copy()
            new_row["paragraph"] = paragraph
            new_row["article_id"] = row["article_id"]
            filtered_rows.append(new_row)

filtered_by_paragraph_df = pd.DataFrame(filtered_rows)


# Count words in each paragraph
filtered_by_paragraph_df["word_count_paragraph"] = filtered_by_paragraph_df["paragraph"].str.split().apply(len)

filtered_by_paragraph_df["word_count_body_text"] = filtered_by_paragraph_df["body_text"].str.split().apply(len)

# Remove paragraphs with more than 400 words for better processing
filtered_by_paragraph_df = filtered_by_paragraph_df[filtered_by_paragraph_df["word_count_paragraph"] <= 400]

# Reset index
filtered_by_paragraph_df.reset_index(drop=True, inplace=True)

# Inspect data
stats = {
    "word_count_paragraph": {
        "max": filtered_by_paragraph_df["word_count_paragraph"].max(),
        "min": filtered_by_paragraph_df["word_count_paragraph"].min(),
        "mean": filtered_by_paragraph_df["word_count_paragraph"].mean()
    },
    "word_count_body_text": {
        "max": filtered_by_paragraph_df["word_count_body_text"].max(),
        "min": filtered_by_paragraph_df["word_count_body_text"].min(),
        "mean": filtered_by_paragraph_df["word_count_body_text"].mean()
    }
}

summary_word_count_df = pd.DataFrame(stats)
summary_word_count_df = summary_word_count_df.T  # transpose for easier reading
summary_word_count_df = summary_word_count_df.round(2)  # round means

print(summary_word_count_df)



# Count unique articles per newspaper
article_counts = filtered_by_paragraph_df.groupby("newspaper")["article_id"].nunique()

# Count paragraphs per newspaper
paragraph_counts = filtered_by_paragraph_df["newspaper"].value_counts()

# Combine into a DataFrame
summary_table = pd.DataFrame({
    "unique_articles": article_counts,
    "paragraphs": paragraph_counts
})

# Add a total row
summary_table.loc["Total"] = [
    filtered_by_paragraph_df["article_id"].nunique(),
    len(filtered_by_paragraph_df)
]

# Display the table
print(summary_table)


#Save results
filtered_by_paragraph_df.to_csv("Hausarbeit2.0/Data/df_newspaper_filtered_by_paragraph_B.csv", index=False)


# Plot wordcount

import matplotlib.pyplot as plt

plt.boxplot(filtered_by_paragraph_df["word_count_paragraph"], vert=False)
plt.xlabel("Paragraphs")
plt.ylabel("Word Count")
plt.title("Boxplot of Paragraph Word Counts")
plt.show()
