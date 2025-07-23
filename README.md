# 🧠 WiSARD: Weightless Neural Networks for Multi-Class Classification

This repository showcases practical, hands-on implementations of the **WiSARD (Wilkie, Stonham and Aleksander's Recognition Device)** weightless neural network model applied to various real-world tabular datasets. The project demonstrates end-to-end workflows—from data ingestion to preprocessing, binarization, model implementation, and evaluation—highlighting transparent, well-documented code designed to attract data science recruiters and ML practitioners.

---

## 🚀 Project Objective

- Implement and demonstrate the **WiSARD weightless neural network model** on prominent UCI datasets.
- Deliver reproducible end-to-end pipelines: from loading and transforming real data to training, testing, and critically interpreting results.
- Highlight the strengths, idiosyncrasies, and pitfalls of weightless neural architectures for structured and categorical data.
- Offer a didactic and recruiter-friendly reference for novel machine learning techniques in tabular data classification.

---

## 📁 Datasets and Notebooks

Each section/notebook implements a *full machine learning workflow* over a specific dataset, including acquisition, preparation, training, and performance reporting. All data are accessed automatically via [`ucimlrepo`](https://github.com/RUB-SysSec/ucimlrepo).

| Dataset                                   | Classes | Description                                               | UCI Link                                                                              | Sample Accuracy |
|--------------------------------------------|---------|-----------------------------------------------------------|---------------------------------------------------------------------------------------|-----------------|
| **Obesity Level Estimation**               | 7       | Predict obesity class from lifestyle and biometric features| [Obesity](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) | 14%            |
| **Car Evaluation**                        | 4       | Assess vehicle acceptability via symbolic attributes       | [Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation)                | 71%            |
| **Contact Lenses**                        | 3       | Recommend lens type from patient characteristics           | [Lenses](https://archive.ics.uci.edu/dataset/58/lenses)                               | 59%            |
| **Student Dropout & Academic Success**     | 3       | Predict undergraduate retention outcomes                   | [Dropout & Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)  | 49%            |
| **Zoo**                                   | 7       | Classify animal species by binary/categorical attributes   | [Zoo](https://archive.ics.uci.edu/dataset/111/zoo)                                    | 13%            |

> **Note:** Each result reflects out-of-the-box WiSARD performance after standardized preprocessing and binarization routines without aggressive hyperparameter tuning.

---

## 🏗️ Project Structure

The core project code (notebook) follows this modular pipeline:

1. **Class Definition: WiSARD**
   - Custom Python implementation with support for variable address sizes and multi-class discrimination.
   - Internally manages binary address generation, RAM discriminators, and class votes.
2. **Data Loading**
   - Uses `ucimlrepo` to fetch open-access datasets seamlessly.
   - Displays variables, schema, and metadata for transparency.
3. **Preprocessing & Binarization**
   - Encodes categorical features using `pd.get_dummies`.
   - Scales data (`StandardScaler`) and binarizes both categorical and continuous features into bit-patterns suitable for WiSARD inputs.
4. **Train/Test Split**
   - Employs stratified splitting (typically 70% test, 30% train) to ensure class balance.
5. **Model Training**
   - Trains the WiSARD classifier by generating RAM addresses and populating class ontologies per input sample.
6. **Inference & Evaluation**
   - Each sample is classified by the model emitting the highest RAM vote.
   - Outputs overall accuracy and, where appropriate, interprets detailed performance per class.
7. **Critical Analysis**
   - Notes on biases, class imbalance, and dataset-model fit.
   - Comments highlight peculiarities (e.g., symbolic data, rule-based vs. learnable patterns).

---

## 💡 Key Features & Innovations

- **Weightless Neural Network Implementation:** Demonstrates a rarely used neural architecture effective for specific classes of problems, especially symbolic learning.
- **Binarization Utility:** Novel functions for bit-level transformation of tabular data—critical for WiSARD.
- **Automated Data Handling:** Every major public UCI dataset used is accessed without manual download/configuration.
- **Full Transparency:** Every code block is annotated, promoting reproducibility and ease of understanding for technical reviewers and hiring managers.
- **Critical Interpretations:** Honest reporting of results—including when WiSARD falls short—offering insight into where classical ML outperforms deep learning and vice versa.

---

## 🧪 Example: Usage (Obesity Dataset)

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

WiSARD model (see included class definition)
class WiSARD:
...
# (Paste implementation from notebook here)

def binarize(data):
# Converts all features to binary string representation
...

1. Fetch dataset
data = fetch_ucirepo(id=544)
X = pd.get_dummies(data.data.features, drop_first=True)
y = LabelEncoder().fit_transform(np.array(data.data.targets).ravel())

2. Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

3. Binarize
X_train_bin = binarize(X_train)
X_test_bin = binarize(X_test)

4. Instantiate and train WiSARD classifier
address_size = max(4, min(8, int(np.log2(X_train_bin.shape))))
model = WiSARD(address_size)
model.train(X_train_bin, y_train)

5. Predict and evaluate
y_pred = model.classify(X_test_bin)
accuracy = np.mean(np.array(y_test) == np.array(y_pred))
print(f"Accuracy: {accuracy:.2%}")

text

---

## 📊 Results Overview

| Dataset                         | Accuracy | Notes                                                                      |
|----------------------------------|----------|---------------------------------------------------------------------------|
| Obesity Level Estimation         | 14%      | Significant class imbalance; many classes, symbolic mix.                   |
| Car Evaluation                   | 71%      | Performs well; symbolic rules are learnable in binary for this dataset.    |
| Contact Lenses                   | 59%      | Small dataset, high rule-logic content, modest performance.                |
| Student Dropout & Academic Success | 49%    | Three-class, imbalanced; moderately challenging, reasonable results.       |
| Zoo                              | 13%      | Poor class separation; unsuitable for WiSARD without further tuning.       |

---

## 🛠️ Technologies

- **Python 3.12**
- **pandas**
- **numpy**
- **scikit-learn**
- **ucimlrepo**

---

## 📌 How to Run

1. Clone the repository.
2. Install dependencies:
    ```
    pip install pandas numpy scikit-learn ucimlrepo
    ```
3. Open and execute `WiSARD.ipynb` (or `WiSARD.py`) in your preferred Jupyter environment.

---

## 🎯 Why This Project Deserves Attention

- **End-to-End Transparency:** Every modeling stage is explained and justified.
- **Didactic & Recruiter-Friendly:** Suitable for portfolio reviews, technical assessments, and interviews.
- **Focus on Rare Models:** Exposes recruiters and technical leads to unconventional neural architectures outside of deep learning mainstream.
- **Clear Strengths & Honest Limits:** Highlights where WiSARD works well—and where it doesn’t.

---

## 👨‍💻 Author

**Thiago Aragão**  
Data Scientist | Deep Learning | Generative AI | NLP | Computer Vision

- GitHub: [@DrAragorn](https://github.com/DrAragorn)
- Email: thiago.alpha.06@gmail.com
- LinkedIn: [linkedin.com/in/thiago-r-aragao](https://linkedin.com/in/thiago-r-aragao)

---

_This repository offers a fresh take on symbolic machine learning with transparent code and open results, aiming to serve as a touchstone for recruiters, data scientists, and anyone interested in weightless neural network models._
