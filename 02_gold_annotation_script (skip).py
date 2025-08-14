import pandas as pd
import os

os.chdir('/home/t/Documents/uni/Master/SoSe25/FP - GenAI')
    
newspaper_df = pd.read_csv('Hausarbeit2.0/Data/df_newspaper_filtered_by_paragraph_mergedAB.csv')


# Script to annotate random sample of 250 paragraphs
ANNOTATED_FILE = 'Hausarbeit2.0/Data/gold standard/gold_annotated.csv'

# If the annotated file doesn't exist yet, create it from a subsample
if not os.path.exists(ANNOTATED_FILE):
    annotated_df = newspaper_df.sample(n=250, random_state=42).reset_index(drop=True)
    annotated_df['gold_standard'] = pd.NA
    annotated_df.to_csv(ANNOTATED_FILE, index=False)
else:
    annotated_df = pd.read_csv(ANNOTATED_FILE)

# Annotation loop
for idx, row in annotated_df.iterrows():
    if pd.isna(row['gold_standard']):
        print("\n" + "-"*80)
        print(f"Paragraph {idx + 1} of {len(annotated_df)}:")
        print(f"📰 Title: {row['title']}")
        print(row['paragraph'])
        print("\nAnnotate sentiment:")
        print("1 = Positive\n2 = Negative\n3 = Neutral/Irrelevant")

        while True:
            user_input = input("Your annotation (1/2/3): ").strip()
            if user_input in {"1", "2", "3"}:
                annotated_df.at[idx, 'gold_standard'] = int(user_input)
                annotated_df.to_csv(ANNOTATED_FILE, index=False)
                break
            else:
                print("Invalid input. Please enter 1, 2, or 3.")

print("\n✅ Annotation complete. Data saved to:", ANNOTATED_FILE)


# Value counts
gold_standard = pd.read_csv('Hausarbeit2.0/Data/gold standard/gold_annotated.csv')

gold_standard['gold_standard'].value_counts()



