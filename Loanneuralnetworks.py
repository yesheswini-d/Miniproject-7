import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ---------------------------------------------------
# 1. Create Dataset
# ---------------------------------------------------

data = pd.DataFrame({

    "Gender": [
        "Male", "Female", "Male", "Male", "Female",
        "Male", "Female", "Male", "Male", "Female",
        "Male", "Female", "Male", "Female", "Male",
        "Male", "Female", "Male", "Male", "Female"
    ],

    "Married": [
        "Yes", "No", "Yes", "Yes", "No",
        "Yes", "No", "Yes", "Yes", "No",
        "Yes", "Yes", "No", "No", "Yes",
        "No", "Yes", "Yes", "No", "Yes"
    ],

    "Education": [
        "Graduate", "Graduate", "Not Graduate", "Graduate",
        "Graduate", "Graduate", "Not Graduate", "Graduate",
        "Graduate", "Graduate", "Not Graduate", "Graduate",
        "Graduate", "Not Graduate", "Graduate", "Graduate",
        "Graduate", "Not Graduate", "Graduate", "Graduate"
    ],

    "Self_Employed": [
        "No", "No", "No", "Yes", "No",
        "No", "Yes", "No", "No", "No",
        "No", "Yes", "No", "No", "Yes",
        "No", "No", "Yes", "No", "No"
    ],

    "ApplicantIncome": [
        5000, 3500, 2800, 6000, 4200,
        7000, 2500, 4500, 8000, 3200,
        3000, 6500, 4000, 2600, 7500,
        3800, 5200, 2900, 4800, 6100
    ],

    "CoapplicantIncome": [
        1500, 0, 1000, 2000, 0,
        2500, 1200, 1000, 3000, 0,
        800, 1800, 0, 500, 2200,
        0, 1200, 900, 0, 1500
    ],

    "LoanAmount": [
        150, 100, 90, 200, 110,
        250, 80, 140, 300, 100,
        90, 180, 120, 75, 220,
        130, 160, 85, 145, 190
    ],

    "Loan_Amount_Term": [
        360, 360, 360, 360, 360,
        360, 360, 360, 360, 360,
        360, 360, 360, 360, 360,
        360, 360, 360, 360, 360
    ],

    "Credit_History": [
        1, 1, 0, 1, 1,
        1, 0, 1, 1, 1,
        0, 1, 1, 0, 1,
        1, 1, 0, 1, 1
    ],

    "Property_Area": [
        "Urban", "Rural", "Rural", "Urban", "Semiurban",
        "Urban", "Rural", "Semiurban", "Urban", "Rural",
        "Rural", "Semiurban", "Urban", "Rural", "Semiurban",
        "Urban", "Semiurban", "Rural", "Urban", "Semiurban"
    ],

    "Loan_Status": [
        1, 1, 0, 1, 1,
        1, 0, 1, 1, 1,
        0, 1, 1, 0, 1,
        1, 1, 0, 1, 1
    ]
})


# ---------------------------------------------------
# 2. Convert Categorical Data
# ---------------------------------------------------

categorical_columns = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area"
]

label_encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    data[column] = encoder.fit_transform(
        data[column]
    )

    label_encoders[column] = encoder


# ---------------------------------------------------
# 3. Separate Features and Target
# ---------------------------------------------------

X = data.drop(
    "Loan_Status",
    axis=1
)

y = data["Loan_Status"]


# ---------------------------------------------------
# 4. Train-Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


# ---------------------------------------------------
# 5. Feature Scaling
# ---------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# ---------------------------------------------------
# 6. Create Neural Network
# ---------------------------------------------------

model = Sequential([

    Dense(
        32,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        8,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ---------------------------------------------------
# 7. Compile Model
# ---------------------------------------------------

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]
)


# ---------------------------------------------------
# 8. Early Stopping
# ---------------------------------------------------

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True
)


# ---------------------------------------------------
# 9. Train Model
# ---------------------------------------------------

history = model.fit(

    X_train,

    y_train,

    epochs=100,

    batch_size=4,

    validation_split=0.2,

    callbacks=[early_stopping],

    verbose=1
)


# ---------------------------------------------------
# 10. Evaluate Model
# ---------------------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nModel Evaluation")
print("----------------")
print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ---------------------------------------------------
# 11. Classification Report
# ---------------------------------------------------

predictions = model.predict(
    X_test,
    verbose=0
)

predictions = (
    predictions > 0.5
).astype(int).flatten()

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ---------------------------------------------------
# 12. Save Model
# ---------------------------------------------------

model.save(
    "loan_model.keras"
)


# ---------------------------------------------------
# 13. Save Scaler and Encoders
# ---------------------------------------------------

with open(
    "preprocessing.pkl",
    "wb"
) as file:

    pickle.dump(
        {
            "scaler": scaler,
            "label_encoders": label_encoders
        },
        file
    )


print("\nModel saved successfully!")

print(
    "loan_model.keras created successfully!"
)

print(
    "preprocessing.pkl created successfully!"
)
