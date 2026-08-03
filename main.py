import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

df = pd.read_csv("C:/Ajay/Forage projects/British Airways/customer_booking.csv", encoding="ISO-8859-1")
print(df.head())
print(df.columns)
print(df.info())
print(df["sales_channel"].unique())

mapping = {
    "Mon": 1,
    "Tue": 2,
    "Wed": 3,
    "Thu": 4,
    "Fri": 5,
    "Sat": 6,
    "Sun": 7,
}
df["flight_day"] = df["flight_day"].map(mapping)
print(df["flight_day"].unique())

print(df.describe())

print((df["booking_complete"].value_counts()))

# Missing values
print(df.isnull().sum())

# Target distribution
print(df["booking_complete"].value_counts())
print(df["booking_complete"].value_counts(normalize=True))

# Categorical columns
print(df.select_dtypes(include=["object", "string"]).columns)

# Numerical columns
print(df.select_dtypes(exclude=["object", "string"]).columns)

df["booking_complete"].value_counts().plot(kind="bar")
plt.title("Booking Completion")
print(plt.show())


#Feature Engineering
#1
df["is_weekend"]=df["flight_day"].isin([6,7]).astype(int)
print(df["is_weekend"])

#2
print(df["flight_duration"].describe())
df["long_haul"]= (df["flight_duration"]>6).astype(int)

#3
df["large_booking"] = (df["num_passengers"] >= 3).astype(int)

#4
df["extra_services"] = (
    df["wants_extra_baggage"] +
    df["wants_preferred_seat"] +
    df["wants_in_flight_meals"]
)

#5
#creating bins

df["booking_category"]= pd.cut(df["purchase_lead"], bins=[0,7,30,90,365], labels=["very late","late","early","very early"])

#Encoding
df = pd.get_dummies(df, drop_first=True)

X = df.drop("booking_complete", axis=1)
y = df["booking_complete"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y #forces both X_train/y_train and X_test/y_test to preserve the same class proportions as the original data

)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test) #model guessing the output for data that is not seen by the model

print("Accuracy:", accuracy_score(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

print(classification_report(y_test, y_pred))

probs = rf.predict_proba(X_test)[:,1]
print("ROC AUC:", roc_auc_score(y_test, probs))

#Cross validation
scores = cross_val_score(
    rf,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print(scores)
print(scores.mean())

#Updated cross validation with StratifiedKFold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf, X, y, cv=skf, scoring="roc_auc")
print(scores)
print(scores.mean())



#Feature Importance
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,8))

importance.head(15).sort_values().plot(kind="barh")

plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

