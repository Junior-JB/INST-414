from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

pd.set_option("display.max_columns", None)

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets.squeeze()

df = pd.concat([X, y], axis=1)

df = df.replace("?", pd.NA)
df = df.dropna()

df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

features = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal"
]

X = df[features].apply(pd.to_numeric)
y = df["num"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=features)
importances = importances.sort_values(ascending=False)

print("\nFeature Importance:\n", importances)

plt.figure()
importances.plot(kind="bar")
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

results = x_test.copy()
results["actual"] = y_test.values
results["predicted"] = y_pred

wrong = results[results["actual"] != results["predicted"]]

print("\nMisclassified Samples:\n", wrong)

fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=wrong.head(5).values,
    colLabels=wrong.columns,
    loc='center'
)

plt.savefig("misclassified_samples.png")
plt.show()

print("\n--- DETAILED MISCLASSIFICATION ANALYSIS ---")

for i, row in wrong.iterrows():
    print("\n--- SAMPLE ---")
    print("Actual:", row["actual"], "| Predicted:", row["predicted"])

    if row["actual"] == 0 and row["predicted"] == 1:
        print("Type: FALSE POSITIVE")
    else:
        print("Type: FALSE NEGATIVE")

    print("Age:", row["age"])
    print("cp:", row["cp"])
    print("chol:", row["chol"])
    print("thalach:", row["thalach"])
    print("oldpeak:", row["oldpeak"])
    print("ca:", row["ca"])
    print("thal:", row["thal"])