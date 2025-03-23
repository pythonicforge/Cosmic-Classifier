# **Planet Classification Model - Code Documentation**

### **Introduction**
This document provides a detailed explanation of the planetary classification model pipeline, including data preprocessing, model architecture, training, evaluation, and inference. Each section of the code is explained to help understand its role in the classification process.

#### **1. Importing Required Libraries**
```python
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
```
**Explanation:**

- **`import pandas as pd`**  
  - Imports the **pandas** library for data manipulation and analysis.  
  - `pd` is a common alias for pandas.  

- **`import tensorflow as tf`**  
  - Imports **TensorFlow**, a deep learning framework.  
  - `tf` is a standard alias for TensorFlow.  

- **`import seaborn as sns`**  
  - Imports **Seaborn**, a visualization library for statistical graphics.  
  - `sns` is a common alias for Seaborn.  

- **`from tensorflow.keras.models import Sequential`**  
  - Imports the **Sequential API** from Keras (which is now part of TensorFlow).  
  - Used to build a neural network layer by layer.  

- **`from tensorflow.keras.layers import Dense, LeakyReLU`**  
  - **`Dense`**: A fully connected neural network layer.  
  - **`LeakyReLU`**: A type of activation function that allows small negative values instead of zero.  

- **`from sklearn.preprocessing import MinMaxScaler`**  
  - Imports **MinMaxScaler** from Scikit-learn.  
  - Used to scale data between a given range (usually 0 to 1).  

- **`from tensorflow.keras.layers import Dropout`**  
  - **Dropout Layer**: A regularization technique that randomly disables some neurons during training to prevent overfitting.  

- **`from tensorflow.keras.layers import BatchNormalization`**  
  - **Batch Normalization**: Helps stabilize and speed up training by normalizing activations in each layer.  

- **`from sklearn.model_selection import train_test_split`**  
  - Splits the dataset into training and testing sets.  

- **`import numpy as np`**  
  - Imports **NumPy**, a library for numerical computing.  
  - `np` is a common alias for NumPy.  

- **`from tensorflow.keras.regularizers import l2`**  
  - Imports **L2 regularization**, which adds a penalty to large weights in the model to reduce overfitting.  

### **2. Loading and Preprocessing Data**
```python
data = pd.read_csv("cosmicclassifierTraining.csv")
scaler = MinMaxScaler(feature_range=(-1, 1))
data.head()
```
**Explanation:**

- **`data = pd.read_csv("cosmicclassifierTraining.csv")`**  
  - Reads a CSV file named **"cosmicclassifierTraining.csv"** into a **pandas DataFrame**.  
  - Assumes the file is in the same directory as the script.  
  - `data` now holds the dataset, which can be manipulated using pandas functions.  

- **`scaler = MinMaxScaler(feature_range=(-1, 1))`**  
  - Creates an instance of **MinMaxScaler** from Scikit-learn.  
  - Scales all feature values **between -1 and 1** (instead of the default 0 to 1).  
  - Helps neural networks converge faster and perform better.  

- **`data.head()`**  
  - Displays the **first five rows** of the dataset.  
  - Helps check if the data has been loaded correctly.  

```python
data = data.dropna(subset=['Prediction'])
data
```
**Explanation:**
- **`data = data.dropna(subset=['Prediction'])`**  
  - Drops any rows in the dataset where the **'Prediction'** column has missing (`NaN`) values.  
  - Ensures the model doesn't train on incomplete data.  
  - The **`subset=['Prediction']`** argument specifies that only the 'Prediction' column should be checked for `NaN` values.  
  - The change is assigned back to `data`, meaning the cleaned dataset replaces the original.

- **`data`**  
  - Displays the modified dataset after dropping rows with missing 'Prediction' values.  
  - Helps verify that `NaN` values were removed successfully.  

This is a **data-cleaning step** to improve model performance.

```python
import re

def category_to_number(val):
    """Converts 'Category_X' to its numeric value X"""
    if isinstance(val, str) and val.startswith("Category_"):
        return int(re.search(r'\d+', val).group())
    return val

data = data.applymap(category_to_number)
data.fillna(data.median(), inplace=True)
```
**Explanation:**
- **`import re`**  
  - Imports the **`re`** (Regular Expressions) module to perform pattern matching on text.  

- **`category_to_number(val)`**  
  - A function to **convert categorical labels** of the form `"Category_X"` into their **numeric values** (`X`).  
  - Uses regex (`re.search(r'\d+', val)`) to extract the number from `"Category_X"`.  
  - If `val` is **not** a string starting with `"Category_"`, it returns `val` unchanged.  

- **`data = data.applymap(category_to_number)`**  
  - Applies `category_to_number` to **each element** of the dataframe.  
  - Ensures categorical features like `"Category_5"` become numerical (`5`), making them usable for ML models.  

- **`data.fillna(data.median(), inplace=True)`**  
  - Fills any **remaining missing values (`NaN`)** with the **median** of each column.  
  - Prevents issues when training the model by ensuring no missing values.  
  - `inplace=True` modifies the dataframe directly.  

#### Why is this important?  
- **Converts categorical data to numeric** → Essential for ML models.  
- **Handles missing values** → Ensures a clean dataset for training.  

```python
data[["Magnetic Field Strength"]] = scaler.fit_transform(data[["Magnetic Field Strength"]])
data[["Atmospheric Density"]] = scaler.fit_transform(data[["Atmospheric Density"]])
data[["Surface Temperature"]] = scaler.fit_transform(data[["Surface Temperature"]])
data[["Gravity"]] = scaler.fit_transform(data[["Gravity"]])
data[["Water Content"]] = scaler.fit_transform(data[["Water Content"]])
data[["Mineral Abundance"]] = scaler.fit_transform(data[["Mineral Abundance"]])
data[["Orbital Period"]] = scaler.fit_transform(data[["Orbital Period"]])
data[["Proximity to Star"]] = scaler.fit_transform(data[["Proximity to Star"]])
data[["Atmospheric Composition Index"]] = scaler.fit_transform(data[["Atmospheric Composition Index"]])
data[["Radiation Levels"]] = scaler.fit_transform(data[["Radiation Levels"]])
```
**Explanation:**
- `scaler.fit_transform(data[["Feature_Name"]])` scales each feature **independently** to the range **(-1, 1)**.
- This **ensures all features have a similar scale**, which helps deep learning models converge faster.

```python
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
```
**Explanation:**

- **`data`** → The full dataset that we are splitting.
- **`test_size=0.25`** → 25% of the data is allocated for **testing**, while **75%** is used for **training**.
- **`random_state=42`** → Ensures **reproducibility**—same split every time.
- **`train_data`** → Contains 75% of the dataset (used to train the model).
- **`test_data`** → Contains 25% of the dataset (used to evaluate the model).

### **3. Drop Prediction Classes**
```python
X_train = train_data.drop("Prediction", axis=1)
y_train = train_data["Prediction"]
X_test = test_data.drop("Prediction", axis=1)
y_test = test_data["Prediction"]
```
**Explanation:**
- Drops the `Prediction` columns and finalizes the data for training and testing

### **4. Defining the Neural Network Model**
```python
model = Sequential()
model.add(Dense(1024,input_shape=(10,), kernel_initializer="glorot_uniform"))
model.add(BatchNormalization())
model.add(Activation(swish))
model.add(Dropout(0.3))

model.add(Dense(512, kernel_initializer="glorot_uniform"))
model.add(BatchNormalization())
model.add(Activation(swish))
model.add(Dropout(0.3))

model.add(Dense(256, kernel_initializer="glorot_uniform"))
model.add(BatchNormalization())
model.add(Activation(swish))
model.add(Dropout(0.3))

model.add(Dense(128, kernel_initializer="glorot_uniform"))
model.add(BatchNormalization())
model.add(Activation(swish))
model.add(Dropout(0.3))

model.add(Dense(64, kernel_initializer="glorot_uniform"))
model.add(BatchNormalization())
model.add(Activation(swish))
model.add(Dropout(0.3))
model.add(Dense(10,activation="softmax"))
model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"])
```
**Explanation:**
- **1024 neurons** → High-capacity feature extraction.  
- **Input shape (10,)** → 10 input features.  
- **Glorot Uniform initializer** → Prevents vanishing/exploding gradients.  
- **BatchNormalization()** → Normalizes activations for stable training.  
- **Swish Activation** → Smooth, non-linear function for better gradient flow.  
- **Dropout (0.3)** → 30% dropout to prevent overfitting.  
- **Layer 1:** 1024 neurons  
- **Layer 2:** 512 neurons  
- **Layer 3:** 256 neurons  
- **Layer 4:** 128 neurons  
- **Layer 5:** 64 neurons  
- **Dense(10, activation="softmax")** → 10-class classification using softmax.  
- **Optimizer: Adam** → Adaptive learning rate for efficient training.  
- **Loss: SparseCategoricalCrossentropy** → Suitable for integer-labeled multi-class problems.  

### **5. Training the Model**
```python
from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor="val_accuracy",patience=10,restore_best_weights=True)
```
**Explanation:**
- Stops training early if validation loss doesn't improve for 10 epochs.

```python
model.fit(X_train, y_train, epochs=1000, batch_size=1024, validation_split=0.2, callbacks=[earlystopping])
```
**Explanation:**
- Trains the model with batch size 1024 for a maximum of 1000 epochs.
- Applies early stopping to prevent overfitting.

### **6. Evaluating the Model**
```python
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy}")
```
**Explanation:**
- Evaluates the model on the test set and prints accuracy.

### **7. Making Predictions on New Data**
```python
testData = pd.read_csv("cosmictest.csv")
results_df = testData.copy()
testData = testData.applymap(category_to_number)
testData.fillna(testData.median(), inplace=True)
```
**Explanation:**
- Preprocess the new data, as we did with the training set

```python
testData[["Magnetic Field Strength"]] = scaler.fit_transform(testData[["Magnetic Field Strength"]])
testData[["Radiation Levels"]] = scaler.fit_transform(testData[["Radiation Levels"]])
testData[["Atmospheric Density"]] = scaler.fit_transform(testData[["Atmospheric Density"]])
testData[["Surface Temperature"]] = scaler.fit_transform(testData[["Surface Temperature"]])
testData[["Gravity"]] = scaler.fit_transform(testData[["Gravity"]])
testData[["Water Content"]] = scaler.fit_transform(testData[["Water Content"]])
testData[["Mineral Abundance"]] = scaler.fit_transform(testData[["Mineral Abundance"]])
testData[["Orbital Period"]] = scaler.fit_transform(testData[["Orbital Period"]])
testData[["Proximity to Star"]] = scaler.fit_transform(testData[["Proximity to Star"]])
testData[["Atmospheric Composition Index"]] = scaler.fit_transform(testData[["Atmospheric Composition Index"]])
```
**Explanation:**
- Normalise the new data in the range (-1, 1)

```python
rest = model.predict(testData)
predicted_labels = np.argmax(rest, axis=1)
print(predicted_labels)
```
**Explanation**
- Makes predictions on the new data and prints the predicted labels

```python
  names = [
      "Bewohnbar",
      "Terraformierbar",
      "Rohstoffreich",
      "Wissenschaftlich",
      "Gasriese",
      "Wüstenplanet",
      "Eiswelt",
      "Toxischetmosäre",
      "Hohestrahlung",
      "Toterahswelt"
  ]
  mapped_array = [names[i] for i in predicted_labels if 0 <= i < len(names)]

  results_df['Predictions'] = predicted_labels
  results_df['Predicted Labels'] = mapped_array

  column_order = [col for col in results_df.columns if col not in ['Predictions', 'Predicted Labels']] + ['Predictions', 'Predicted Labels']
  results_df = results_df[column_order]

  results_df.to_csv('model_predictions.csv', index=False)

  print("Predictions saved to 'model_predictions.csv'")
```
**Explanation**
Finally, we create a new csv file named `model_predictions.csv` where we store the predictions (integer classes) and the predicted classes.

### **Conclusion**
This planetary classification model effectively preprocesses noisy data, normalizes features, and leverages a deep neural network to achieve high accuracy in predicting planetary types. By implementing advanced techniques such as batch normalization, dropout, and the Swish activation function, the model ensures robust performance while mitigating overfitting.

Future improvements could include fine-tuning hyperparameters, experimenting with alternative architectures, and integrating additional planetary features for enhanced classification accuracy. This project serves as a strong foundation for further exploration in astronomical data classification and machine learning applications in astrophysics.

