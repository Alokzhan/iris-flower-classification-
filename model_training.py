{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "710aaffa-81c5-4942-892f-1a6978fd8ef8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Model and encoder saved successfully!\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.datasets import load_iris\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "import pickle\n",
    "\n",
    "# Load Dataset\n",
    "iris = load_iris()\n",
    "data = pd.DataFrame(data=iris.data, columns=iris.feature_names)\n",
    "data['species'] = iris.target\n",
    "data['species'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})\n",
    "\n",
    "# Prepare Data\n",
    "X = data[iris.feature_names]\n",
    "y = data['species']\n",
    "\n",
    "le = LabelEncoder()\n",
    "y_encoded = le.fit_transform(y)\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)\n",
    "\n",
    "# Train Model\n",
    "model = LogisticRegression(max_iter=200)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Save model and encoder\n",
    "with open('iris_model.pkl', 'wb') as f:\n",
    "    pickle.dump(model, f)\n",
    "\n",
    "with open('label_encoder.pkl', 'wb') as f:\n",
    "    pickle.dump(le, f)\n",
    "\n",
    "print(\"✅ Model and encoder saved successfully!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "09c1b415-92ea-4779-a14d-bbc7874bdf0a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "705f85fe-c1fc-4ccf-bf67-07687ecfcda3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
