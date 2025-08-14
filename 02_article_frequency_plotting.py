import pandas as pd
import os
os.chdir('/home/t/Documents/uni/Master/SoSe25/FP - GenAI')

# Load datasets A and B
newspaper_df_A = pd.read_csv('Hausarbeit2.0/Data/df_newspaper_filtered_by_paragraph_A.csv')
newspaper_df_B = pd.read_csv('Hausarbeit2.0/Data/df_newspaper_filtered_by_paragraph_B.csv')


# Ensure article IDs are unique across groups
newspaper_df_A['article_id'] = 'A_' + newspaper_df_A['article_id'].astype(str)
newspaper_df_B['article_id'] = 'B_' + newspaper_df_B['article_id'].astype(str)


# Merge the dataframes
merged_df = pd.concat([newspaper_df_A, newspaper_df_B], ignore_index=True)


# Drop unwanted columns
columns_to_drop = ['Unnamed: 0', 'text', 'group', 'word_count_paragraph', 'word_count_body_text']
merged_df = merged_df.drop(columns=columns_to_drop, errors='ignore')

# Save to CSV
merged_df.to_csv('Hausarbeit2.0/Data/df_newspaper_filtered_by_paragraph_mergedAB.csv', index=False)



# Plot article frequency over time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Ensure 'date' is datetime
merged_df['date'] = pd.to_datetime(merged_df['date'])

# Drop duplicate articles (if articles have multiple paragraphs)
unique_articles = merged_df.drop_duplicates(subset=['article_id'])

# Define filters for newspapers
filters = {
    "Welt": unique_articles['newspaper'] == 'Welt',
    "FR": unique_articles['newspaper'] == 'FR',
}

# Colors for newspapers
color_map = {
    "Welt": "#034158",  # dark blue
    "FR":   "#3f6224",  # light green
}


# Prepare frequency data per newspaper, dropping the last month
frequency_dfs = []
for label, condition in filters.items():
    df_filtered = unique_articles[condition]
    freq = df_filtered.groupby(df_filtered['date'].dt.to_period('M')).size()
    freq.index = freq.index.to_timestamp()
    freq = freq.iloc[:-1]
    frequency_dfs.append(freq.rename(label))


# Combine frequencies into one DataFrame (with last month removed)
freq_df = pd.concat(frequency_dfs, axis=1)

plt.figure(figsize=(16, 4.5))

# Plot each newspaper's frequency
for label in freq_df.columns:
    plt.plot(freq_df.index, freq_df[label], label=label, color=color_map[label], linewidth=1.5)

# Arithmetic mean line
arithmetic_mean = freq_df.mean(axis=1)
plt.plot(arithmetic_mean.index, arithmetic_mean.values,
         label="Arithmetic Mean", color="black", linestyle="--", linewidth=2)

# Set x-axis ticks and formatting explicitly here:
ax = plt.gca()  # get current axis

# Set major locator to every month
ax.xaxis.set_major_locator(mdates.MonthLocator())

# Set major formatter to Year-Month string
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Rotate tick labels for readability
plt.xticks(rotation=45)

# Set x-axis limits to span full range of data
ax.set_xlim(freq_df.index.min(), freq_df.index.max())

# Draw vertical lines and add event labels (skip events beyond last month)
events = {
    pd.to_datetime("2023-11-01"): "\n\n FEG 1",
    pd.to_datetime("2024-03-01"): "\n\n FEG 2",
    pd.to_datetime("2024-06-01"): "\n\n FEG 3",
    pd.to_datetime("2024-09-01"): " Sept 2024:\n Munich shooting\n and new border controls",
    pd.to_datetime("2025-02-01"): "\n\n Feb 2025: General elections"
}

for date, label in events.items():
    if date <= freq_df.index.max():  # Only plot if event before last month removed
        plt.axvline(x=date, color='black', linestyle='-', linewidth=0.8)
        plt.text(date, plt.ylim()[1] * 0.2, label, fontsize=9, va='top', ha='left')

# Final plot formatting
plt.title("Article Frequency Over Time by Newspaper")
plt.xlabel("Date")
plt.ylabel("Number of Articles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Hausarbeit2.0/Plots/Articles_per_month_newspaper.pdf", format='pdf', bbox_inches='tight')
plt.show()



# Count unique articles per newspaper
article_counts = merged_df.groupby("newspaper")["article_id"].nunique()

# Count paragraphs per newspaper
paragraph_counts = merged_df["newspaper"].value_counts()

# Calculate percentages
article_percent = (article_counts / article_counts.sum()) * 100
paragraph_percent = (paragraph_counts / paragraph_counts.sum()) * 100

# Combine into a percentage summary table
summary_table_pct = pd.DataFrame({
    "unique_articles (%)": article_percent,
    "paragraphs (%)": paragraph_percent
})

# Optional: Add a total row (will just be 100%)
summary_table_pct.loc["Total"] = [100.0, 100.0]

# Round for cleaner display
summary_table_pct = summary_table_pct.round(2)

# Display the table
print(summary_table_pct)



# Count words in each paragraph and article
merged_df["word_count_paragraph"] = merged_df["paragraph"].str.split().apply(len)

merged_df["word_count_body_text"] = merged_df["body_text"].str.split().apply(len)

stats = {
    "word_count_paragraph": {
        "max": merged_df["word_count_paragraph"].max(),
        "min": merged_df["word_count_paragraph"].min(),
        "mean": merged_df["word_count_paragraph"].mean()
    },
    "word_count_body_text": {
        "max": merged_df["word_count_body_text"].max(),
        "min": merged_df["word_count_body_text"].min(),
        "mean": merged_df["word_count_body_text"].mean()
    }
}

summary_word_count_df = pd.DataFrame(stats)
summary_word_count_df = summary_word_count_df.T  # transpose for easier reading
summary_word_count_df = summary_word_count_df.round(2)  # round means

print(summary_word_count_df)



# Explore articles in September/October 2024: 

# Convert 'month' to datetime
merged_df['month'] = pd.to_datetime(merged_df['month'], errors='coerce')

# Filter for September 2024
filtered_df = merged_df[
    (merged_df['month'].dt.month == 9) &
    (merged_df['month'].dt.year == 2024)
].reset_index(drop=True)

# Show title and paragraph one by one
for idx, row in filtered_df.iterrows():
    input(f"\nPress Enter to read paragraph {idx + 1}/{len(filtered_df)}:\n")
    print(f"Title: {row['title']}\n")
    print(row['paragraph'])
