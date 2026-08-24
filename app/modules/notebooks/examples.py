EXAMPLE_NOTEBOOKS = {
    "python-basics": {
        "title": "Python Basics",
        "description": "Variables, collections, functions, and control flow.",
        "category": "Python",
        "cells": [
            ("markdown", "# Python basics\nA small, CPU-only introduction."),
            ("code", "values = [1, 2, 3, 4]\nprint([value ** 2 for value in values])"),
        ],
    },
    "pandas-analysis": {
        "title": "pandas Data Analysis",
        "description": "Create, summarize, and filter a DataFrame.",
        "category": "Data",
        "cells": [
            ("markdown", "# pandas data analysis"),
            (
                "code",
                "import pandas as pd\ndf = pd.DataFrame({'team': ['A', 'A', 'B'], 'score': [8, 12, 10]})\ndf",
            ),
            ("code", "df.groupby('team', as_index=False)['score'].mean()"),
        ],
    },
    "matplotlib-visualization": {
        "title": "Matplotlib Visualization",
        "description": "Render a small line chart inline.",
        "category": "Visualization",
        "cells": [
            ("markdown", "# Matplotlib visualization"),
            (
                "code",
                "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3], [1, 4, 9], marker='o')\nplt.title('Squares')\nplt.show()",
            ),
        ],
    },
    "sklearn-classification": {
        "title": "scikit-learn Classification",
        "description": "Train a compact iris classifier without downloads.",
        "category": "Machine Learning",
        "cells": [
            (
                "code",
                "from sklearn.datasets import load_iris\nfrom sklearn.ensemble import RandomForestClassifier\nX, y = load_iris(return_X_y=True)\nmodel = RandomForestClassifier(n_estimators=20, random_state=42).fit(X, y)\nprint(model.score(X, y))",
            ),
        ],
    },
    "sklearn-clustering": {
        "title": "scikit-learn Clustering",
        "description": "Cluster a deterministic toy dataset.",
        "category": "Machine Learning",
        "cells": [
            (
                "code",
                "from sklearn.cluster import KMeans\nX = [[0, 0], [0, 1], [9, 9], [10, 9]]\nlabels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X)\nprint(labels)",
            ),
        ],
    },
    "pytorch-neural-network": {
        "title": "PyTorch Neural Network",
        "description": "A tiny CPU tensor and linear layer example.",
        "category": "Deep Learning",
        "cells": [
            (
                "code",
                "import torch\ntorch.manual_seed(42)\nlayer = torch.nn.Linear(2, 1)\nprint(layer(torch.tensor([[1.0, 2.0]])))",
            ),
        ],
    },
    "nlp-sentiment": {
        "title": "NLP Text Classification",
        "description": "A lightweight bag-of-words sentiment workflow.",
        "category": "NLP",
        "cells": [
            (
                "code",
                "from sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.linear_model import LogisticRegression\ntexts = ['great product', 'very helpful', 'bad result', 'not useful']\ny = [1, 1, 0, 0]\nvectorizer = CountVectorizer()\nmodel = LogisticRegression(random_state=42).fit(vectorizer.fit_transform(texts), y)\nprint(model.predict(vectorizer.transform(['great and helpful'])))",
            ),
        ],
    },
}
