import pandas as pd

reviews = []
sentiments = []

with open("test.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split("\t")   # try tab split

        if len(parts) == 2:
            reviews.append(parts[0])
            sentiments.append(parts[1])

# Create DataFrame
df = pd.DataFrame({
    "review": reviews,
    "sentiment": sentiments
})

# Save as CSV
df.to_csv("dataset.csv", index=False)

print("Converted to CSV successfully!")