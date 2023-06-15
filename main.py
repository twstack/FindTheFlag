import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# 1. Load data
flags = pd.read_csv('flags.csv', header=0)

# 2. Print columns and head of DataFrame
print(flags.columns)
print(flags.head())

# 3. Create labels
labels = flags[['Landmass']]

# 4. Add all features related to shapes
data = flags[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange",
              "Circles","Crosses","Saltires","Quarters","Sunstars",
              "Crescent","Triangle"]]

# 5. Create the tree
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)
scores = []
for i in range(1, 21):
    tree = DecisionTreeClassifier(random_state=1, max_depth=i)
    tree.fit(train_data, train_labels)
    scores.append(tree.score(test_data, test_labels))

# 6. Plot the scores
plt.plot(range(1, 21), scores)
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.show()
