"""Small, deterministic starter curriculum. Add topics here without changing the API."""
from __future__ import annotations

LABS = {
    "automl": ("Machine Learning Mastery", "/automl", "ML"),
    "autodl": ("Deep Learning Mastery", "/autodl", "DL"),
    "autonlp": ("NLP Mastery", "/autonlp", "NLP"),
    "python": ("Python Mastery", "/python-lab", "Python"),
    "sql": ("SQL Mastery", "/sql", "SQL"),
    "eda": ("Data Analysis Mastery", "/eda", "EDA"),
    "genai": ("Generative AI Mastery", "/genai", "GenAI"),
    "prompting": ("Prompt Engineering Mastery", "/genai?tab=prompts", "Prompt Lab"),
}


def topic(module: str, key: str, title: str, level: str, theory: str,
          example: str, mistake: str, question: str, choices: list[str],
          answer: int, concepts: list[str], *, stage: str = "Foundations",
          activity: str = "reasoning", prerequisite: str | None = None) -> dict:
    return {
        "id": f"{module}-{key}", "module": module, "title": title,
        "level": level, "stage": stage, "prerequisites": [prerequisite] if prerequisite else [],
        "description": theory, "theory_sections": [{"heading": title, "text": theory}],
        "example": example, "common_mistake": mistake,
        "key_takeaways": concepts, "practice_activity": activity,
        "coding_exercise": "Open the related lab and reproduce the example." if module in {"python", "sql", "automl", "autodl", "autonlp"} else None,
        "case_study": None, "assessment_criteria": concepts,
        "interview_competencies": concepts,
        "quick_check": {"question": question, "choices": choices, "answer_index": answer},
        "lab_url": LABS[module][1],
    }


TOPICS = [
    topic("automl", "features-target", "Features and target", "FOUNDATION", "Each row is an observation. Features are inputs; the target is the outcome to predict.", "For stroke risk, age and glucose are features; stroke is the target.", "Including a post-diagnosis field leaks the answer.", "Which column should a stroke classifier predict?", ["Age", "Stroke outcome", "Patient ID"], 1, ["features", "target", "leakage"]),
    topic("automl", "splits", "Train, validation, and test", "BEGINNER", "Fit on training data and reserve separate data for evaluation. Never learn preprocessing statistics from held-out rows.", "Fit a scaler on training rows, then transform test rows.", "Evaluating on training data overstates quality.", "Which data should stay unseen during fitting?", ["Training", "Held-out test", "All rows"], 1, ["split", "generalization", "leakage"], stage="Data Preparation", prerequisite="automl-features-target"),
    topic("automl", "imbalance", "Classification under imbalance", "INTERMEDIATE", "Accuracy alone can conceal missed positives. Inspect recall, precision, confusion matrix, ROC and PR curves.", "A stroke model with 97% accuracy but 30% recall misses many cases.", "High accuracy does not guarantee useful recall.", "Which metric directly measures detected positives?", ["Recall", "Training time", "MAE"], 0, ["recall", "false negatives", "precision"], stage="Classification", prerequisite="automl-splits"),
    topic("automl", "regression-errors", "Regression errors", "INTERMEDIATE", "MAE measures average absolute error; RMSE emphasizes large errors; R² compares to a mean baseline.", "A large RMSE relative to MAE suggests some large misses.", "R² is not an error in target units.", "Which metric penalizes large errors more strongly?", ["MAE", "RMSE", "Accuracy"], 1, ["mae", "rmse", "residual"], stage="Regression", prerequisite="automl-splits"),
    topic("automl", "clustering", "Clustering and choosing K", "INTERMEDIATE", "Clustering groups similar observations without labels. Compare separation and cluster profiles before naming groups.", "KMeans groups customers; silhouette helps compare candidate K values.", "An elbow does not prove a unique optimal K.", "What does silhouette describe?", ["Separation and cohesion", "Classification recall", "Forecast horizon"], 0, ["kmeans", "silhouette", "profiles"], stage="Clustering", prerequisite="automl-splits"),
    topic("autodl", "neurons", "Neurons and gradients", "FOUNDATION", "A neuron combines weighted inputs with a bias and activation. Gradients show how loss changes with parameters.", "Backpropagation computes gradients; an optimizer updates weights.", "An epoch is not one gradient update unless there is one batch.", "What updates model weights?", ["Optimizer", "Activation alone", "Test split"], 0, ["weights", "loss", "gradient"]),
    topic("autodl", "cnn", "CNN image classification", "BEGINNER", "Convolutions reuse filters across an image. Separate validation and independent test data reveal different risks.", "A plant-disease CNN learns local image patterns.", "A high training score alone does not prove robustness.", "Which data best checks final generalization?", ["Training", "Independent test", "Augmented copies"], 1, ["convolution", "validation", "test"], stage="Computer Vision", prerequisite="autodl-neurons"),
    topic("autodl", "pytorch", "PyTorch training loop", "INTERMEDIATE", "A small loop runs forward, loss, zero_grad, backward, and optimizer.step before evaluation.", "Use CPU tensors and a tiny local dataset in Python Lab.", "Forgetting zero_grad accumulates gradients unintentionally.", "What must happen before backward in a typical loop?", ["zero_grad", "Model export", "Test prediction"], 0, ["forward", "loss", "optimizer"], stage="Coding", prerequisite="autodl-neurons", activity="coding"),
    topic("autonlp", "tfidf", "Bag of Words and TF-IDF", "FOUNDATION", "Count vectors represent term frequency; TF-IDF lowers the weight of widespread terms.", "A support-ticket classifier learns from local labeled text.", "Word counts do not capture every nuance of meaning.", "What does TF-IDF downweight?", ["Common terms", "Every rare term", "Class labels"], 0, ["vocabulary", "tfidf", "classification"]),
    topic("autonlp", "metrics", "Text classification evaluation", "BEGINNER", "Separate training and held-out texts. Inspect per-class precision, recall, and confusion patterns.", "A spam filter may trade missed spam for false alarms.", "A tiny random split can produce unstable estimates.", "Which metric highlights missed positive texts?", ["Recall", "Vocabulary size", "Latency"], 0, ["recall", "held-out", "labels"], stage="Evaluation", prerequisite="autonlp-tfidf"),
    topic("autonlp", "transformers", "When is a transformer needed?", "INTERMEDIATE", "Start with a simple baseline. Choose a transformer when context-sensitive meaning justifies extra data and compute.", "Compare TF-IDF logistic regression with a transformer on the same held-out set.", "A larger model is not automatically better for a small task.", "What should precede a transformer decision?", ["A baseline and task evaluation", "More layers", "No test set"], 0, ["baseline", "context", "tradeoff"], stage="Advanced", prerequisite="autonlp-metrics"),
    topic("python", "basics", "Values and control flow", "FOUNDATION", "Variables name values; conditions select branches; loops repeat work. Use readable names and small functions.", "for customer in customers: print(customer['name'])", "A variable name should describe its meaning, not its type alone.", "What runs code repeatedly?", ["Loop", "Comment", "Type hint"], 0, ["variables", "conditions", "loops"], activity="coding"),
    topic("python", "collections", "Collections and functions", "BEGINNER", "Lists preserve order, sets remove duplicates, dictionaries map keys to values, and functions package reusable logic.", "def high_spend(rows, limit): return [r for r in rows if r['spend'] > limit]", "A mutable default argument can leak state across calls.", "Which collection maps keys to values?", ["List", "Dictionary", "Tuple"], 1, ["list", "dictionary", "function"], stage="Intermediate", prerequisite="python-basics", activity="coding"),
    topic("python", "pandas", "pandas grouping and cleaning", "INTERMEDIATE", "Inspect types and missingness before filtering, grouping, or joining tables.", "sales.groupby('region')['amount'].sum()", "A groupby object needs an aggregation to produce totals.", "Which operation totals amount by region?", ["groupby then sum", "head only", "dropna only"], 0, ["groupby", "missingness", "join"], stage="Data Python", prerequisite="python-collections", activity="coding"),
    topic("sql", "select", "SELECT, WHERE, and ORDER BY", "FOUNDATION", "SELECT chooses columns, WHERE filters rows, and ORDER BY sorts the result. LIMIT bounds displayed rows.", "SELECT name FROM customers WHERE city = 'Chennai' ORDER BY name;", "WHERE filters before grouping; HAVING filters groups.", "Which clause filters individual rows?", ["WHERE", "HAVING", "ORDER BY"], 0, ["select", "where", "order by"], activity="coding"),
    topic("sql", "groups", "GROUP BY and HAVING", "BEGINNER", "Aggregate values within groups; use HAVING for aggregate conditions.", "SELECT department, COUNT(*) AS employee_count FROM employees GROUP BY department HAVING COUNT(*) >= 1;", "Selecting an ungrouped, unaggregated column can be invalid.", "Which clause filters aggregated groups?", ["WHERE", "HAVING", "LIMIT"], 1, ["group by", "sum", "having"], stage="Intermediate", prerequisite="sql-select", activity="coding"),
    topic("sql", "joins", "Joins and CTEs", "INTERMEDIATE", "Join related tables on keys; a CTE names an intermediate query for clarity.", "SELECT c.name, SUM(p.price * o.quantity) AS total_revenue FROM customers c JOIN orders o ON o.customer_id = c.id JOIN products p ON p.id = o.product_id GROUP BY c.name;", "An incomplete join key can multiply rows.", "Which join preserves unmatched left rows?", ["INNER JOIN", "LEFT JOIN", "CROSS JOIN"], 1, ["join", "keys", "cte"], stage="Intermediate", prerequisite="sql-groups", activity="coding"),
    topic("eda", "schema", "Understand a dataset", "FOUNDATION", "Inspect row count, columns, data types, units, and candidate targets before charting.", "Check whether salary is numeric and department is categorical.", "A numeric identifier is not automatically a meaningful measure.", "What should be inspected before analysis?", ["Schema and missingness", "Only colors", "Only row order"], 0, ["schema", "types", "target"]),
    topic("eda", "quality", "Missing values and duplicates", "BEGINNER", "Measure missingness and duplicates before choosing a cleaning strategy; preserve evidence of changes.", "Count null salary values and duplicate customer IDs.", "Blindly dropping rows may bias results.", "What should happen before dropping missing rows?", ["Measure impact", "Ignore columns", "Train immediately"], 0, ["missing", "duplicates", "cleaning"], stage="Data Quality", prerequisite="eda-schema"),
    topic("eda", "charts", "Choose an informative chart", "INTERMEDIATE", "Match the chart to variable types and comparison goal: boxplots compare numeric distributions across categories.", "Compare salary distributions by department with a boxplot.", "A line chart implies order and should not connect unrelated categories.", "Which chart compares salary spread by department?", ["Boxplot", "Pie chart", "ROC curve"], 0, ["chart selection", "distribution", "interpretation"], stage="Visualization", prerequisite="eda-quality"),
    topic("genai", "llm", "LLM fundamentals", "FOUNDATION", "An LLM predicts tokens from context. Output can be fluent yet unsupported, so verify claims and sources.", "Ask for a concise answer with explicit uncertainty when evidence is missing.", "Fluency is not evidence of truth.", "What limits how much context a model can inspect?", ["Context window", "Chart width", "Batch size"], 0, ["tokens", "context", "hallucination"]),
    topic("genai", "rag", "Document-grounded generation", "BEGINNER", "RAG retrieves relevant document passages, then asks the model to answer from those passages with citations.", "Select research papers, retrieve matching chunks, and cite the cited page.", "A retrieved passage is not proof that every sentence in an answer is supported.", "What should happen when evidence is missing?", ["State insufficiency", "Invent a citation", "Search unrelated files"], 0, ["retrieval", "grounding", "citations"], stage="RAG", prerequisite="genai-llm"),
    topic("genai", "agents", "Tool-use and agents", "INTERMEDIATE", "An agent decides when a tool is needed, supplies complete arguments, and handles failures. Tools can affect external state.", "A data question may require a database query; a writing task may not.", "Invoking every tool for every question adds risk and cost.", "What belongs in a tool request?", ["Required arguments", "Hidden API key", "Invented result"], 0, ["tools", "arguments", "failure"], stage="Agents", prerequisite="genai-rag"),
    topic("prompting", "clarity", "Clear instructions and constraints", "FOUNDATION", "Name the task, audience, limits, and desired output so success can be checked.", "Explain precision to a first-year student in three bullets with one example.", "'Analyze this' leaves objective and format unclear.", "What improves an underspecified prompt?", ["Audience and output format", "More vague adjectives", "No task"], 0, ["task", "audience", "format"], activity="prompt"),
    topic("prompting", "grounding", "Grounded and RAG prompts", "BEGINNER", "Specify allowed documents, request citations, and instruct the model to say when evidence is insufficient.", "Use selected project documents only; cite each claim or state that evidence is missing.", "A source-only instruction without selected documents cannot provide grounding.", "What should a grounded prompt include?", ["Source scope and refusal", "Only a role", "No citations"], 0, ["scope", "citations", "refusal"], stage="Grounding", prerequisite="prompting-clarity", activity="prompt"),
    topic("prompting", "compare", "Prompt A/B reasoning", "INTERMEDIATE", "Compare two prompt outputs against the same goal and explicit criteria; do not assume the longer prompt wins.", "Judge accuracy, completeness, format, and latency across saved runs.", "Choosing a winner by style alone hides factual errors.", "What makes an A/B comparison fair?", ["Same task and criteria", "Different tasks", "Only length"], 0, ["criteria", "comparison", "evidence"], stage="Evaluation", prerequisite="prompting-clarity", activity="prompt"),
    topic("automl", "preparation", "Data preparation and leakage", "BEGINNER", "Impute and encode using training data only. Exclude identifiers that do not generalize and check for leaked target information.", "A post-outcome treatment column must not predict the earlier diagnosis.", "Fitting preprocessing on all rows leaks test information.", "When should imputation values be learned?", ["Training split only", "Test split only", "After seeing evaluation"], 0, ["imputation", "encoding", "leakage"], stage="Data Preparation", prerequisite="automl-splits"),
    topic("automl", "explainability", "Feature importance and limitations", "ADVANCED", "Feature importance describes the fitted model's reliance on inputs; it is not a causal claim. Check stability and correlated features.", "A high age importance does not prove changing age would change an outcome.", "Importance is not causation.", "What does model feature importance establish?", ["Model influence", "Causation", "Fairness certification"], 0, ["importance", "causation", "correlation"], stage="Advanced ML", prerequisite="automl-imbalance"),
    topic("python", "files-errors", "Files, exceptions, and modules", "INTERMEDIATE", "Use context managers for files, catch specific exceptions, and import reusable standard-library modules.", "with open(path) as file: text = file.read()", "A broad bare except can hide programming errors.", "What closes a file reliably?", ["Context manager", "Global variable", "Print call"], 0, ["files", "exceptions", "modules"], stage="Intermediate", prerequisite="python-collections", activity="coding"),
    topic("python", "models", "Python model workflow", "INTERMEDIATE", "Separate features and target, split held-out data, fit preprocessing on training rows, then evaluate the model.", "A sklearn Pipeline keeps preprocessing together with an estimator.", "Testing on the fitted training rows overstates generalization.", "Which split should report final performance?", ["Held-out test", "Training", "Neither"], 0, ["pipeline", "split", "metrics"], stage="AI Python", prerequisite="python-pandas", activity="coding"),
    topic("sql", "windows", "Window functions", "ADVANCED", "Window functions compute across related rows without collapsing each row. PARTITION BY forms groups and ORDER BY defines sequence.", "ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount DESC)", "GROUP BY removes row detail that window functions preserve.", "Which clause groups rows inside a window?", ["PARTITION BY", "HAVING", "LIMIT"], 0, ["window", "partition", "rank"], stage="Advanced", prerequisite="sql-joins", activity="coding"),
    topic("sql", "revenue", "Business revenue query", "INTERMEDIATE", "Join orders to customers on keys, group by the business dimension, and check row counts before trusting totals.", "Monthly revenue uses a date bucket and SUM(order_amount).", "Duplicate join matches can inflate revenue.", "What should be checked after a join?", ["Row cardinality", "Only column color", "Model accuracy"], 0, ["joins", "aggregation", "validation"], stage="Business SQL", prerequisite="sql-joins", activity="coding"),
    topic("eda", "relationships", "Correlations and comparisons", "INTERMEDIATE", "A correlation summarizes a numeric relationship, but outliers and confounders can mislead. Use plots alongside summary statistics.", "Compare price and area with a scatter plot before interpreting correlation.", "Correlation does not establish causality.", "What should accompany a correlation coefficient?", ["A relevant plot and context", "Only a pie chart", "No data check"], 0, ["correlation", "scatter", "outliers"], stage="Bivariate", prerequisite="eda-charts"),
    topic("eda", "insights", "Communicating an insight", "ADVANCED", "Explain the observed pattern, uncertainty, business relevance, and recommended next check.", "Sales rose in two regions, but missing store data may affect the comparison.", "A chart without context is not a complete business conclusion.", "What makes an insight actionable?", ["Evidence, limits, and next step", "Only a title", "Only a color"], 0, ["interpretation", "uncertainty", "communication"], stage="Insight", prerequisite="eda-relationships"),
]

# Static, concise AutoML study units. These explain concepts; they do not alter AutoML training.
ML_DEPTH = [
    ("foundations-depth", "ML foundations", "FOUNDATION", "ML Foundations", "automl-features-target", [
        ("AI versus ML", "AI is the broader goal of intelligent behavior; ML learns patterns from data."),
        ("Features and target", "Features are inputs; a supervised target is the known outcome."),
        ("Supervised versus unsupervised", "Classification and regression learn from labels; clustering discovers groups without labels."),
        ("Train, validation, test", "Fit on train, choose settings on validation, and reserve test for final evaluation."),
        ("Overfit and underfit", "Overfit models memorize training details; underfit models miss useful patterns."),
        ("Bias and variance", "Simple models may have high bias; sensitive models can have high variance."),
        ("Leakage", "Post-outcome columns or preprocessing fitted on test rows make scores unreliable.")],
     "Confusing high training accuracy with held-out quality", "Which split should remain untouched until final evaluation?", ["Test", "Train", "All rows"], 0,
     "Identify a possible leaked column in a medical dataset.", "What evidence would justify releasing a model?"),
    ("preparation-depth", "Data preparation decisions", "BEGINNER", "Data Preparation", "automl-preparation", [
        ("Missing values", "Measure missingness; impute using training data and consider whether absence itself matters."),
        ("Encoding", "Categorical values require a consistent numeric representation across train and test."),
        ("Scaling", "Distance-based and regularized methods often benefit from training-fitted scaling."),
        ("Identifiers", "IDs may memorize rows without explaining future outcomes; exclude them when appropriate."),
        ("Outliers", "Check whether extremes are errors or real observations before removing them."),
        ("Class imbalance", "Count classes and inspect minority-class precision and recall, not accuracy alone."),
        ("Feature selection", "Choose features using training evidence to avoid test-set leakage.")],
     "Dropping every row with a missing value without measuring impact", "When must an imputer learn its values?", ["Training data", "Test data", "All data"], 0,
     "Inspect missingness and class counts before training.", "How would you prevent preprocessing leakage?"),
    ("classification-depth", "Classification methods and diagnostics", "INTERMEDIATE", "Classification", "automl-imbalance", [
        ("Logistic Regression", "A linear classifier estimates class probability from weighted features."),
        ("Decision Tree", "A tree partitions data with feature-based rules; deep trees can overfit."),
        ("Random Forest", "An ensemble averages many trees to reduce variance."),
        ("KNN", "K-nearest neighbors predicts from nearby examples; scaling affects distances."),
        ("SVM", "Support vector machines seek a separating boundary; kernels can model nonlinear patterns."),
        ("Naive Bayes", "A probabilistic baseline uses a conditional-independence assumption."),
        ("Gradient Boosting", "Sequential trees correct earlier errors; tuning controls complexity."),
        ("Accuracy and confusion matrix", "Accuracy is total correct divided by total; the matrix exposes each error type."),
        ("Precision, recall, F1", "Precision measures reliability of positive predictions; recall measures found positives; F1 balances both."),
        ("ROC-AUC and PR", "ROC studies true-positive versus false-positive rates; PR is useful for rare positives."),
        ("Threshold and imbalance", "Changing a decision threshold trades false alarms against misses without retraining." )],
     "Using accuracy alone for a rare positive class", "Which metric emphasizes finding positive cases?", ["Recall", "MAE", "R²"], 0,
     "Compare two classifiers using confusion matrices and PR curves.", "When would you prioritize recall over precision?"),
    ("regression-depth", "Regression methods and residuals", "INTERMEDIATE", "Regression", "automl-regression-errors", [
        ("Linear Regression", "Fits an additive linear relationship between features and a numeric target."),
        ("Ridge and Lasso", "Regularization penalizes coefficients; Lasso can set some to zero."),
        ("Trees and ensembles", "Trees capture nonlinear splits; ensembles combine learners for stronger generalization."),
        ("R²", "Compares squared error to predicting the test-set mean; it can be negative."),
        ("MAE and MSE", "MAE is average absolute error; MSE squares errors and emphasizes large misses."),
        ("RMSE and MAPE", "RMSE returns to target units; MAPE is unstable near zero actual values."),
        ("Residuals", "Plot actual minus predicted against predictions to spot bias or changing error spread.")],
     "Calling a low training error proof of generalization", "Which error metric emphasizes large misses?", ["MSE", "MAE", "Accuracy"], 0,
     "Inspect residual patterns and compare MAE with RMSE.", "Why can MAPE mislead when actual values approach zero?"),
    ("clustering-depth", "Clustering methods and interpretation", "INTERMEDIATE", "Clustering", "automl-clustering", [
        ("KMeans", "Assigns rows to K centroids; compare several K values rather than guessing."),
        ("DBSCAN", "Groups dense regions and labels noise; standard DBSCAN does not predict unseen rows."),
        ("Hierarchical clustering", "Builds nested merges that can be cut into a chosen number of clusters."),
        ("Silhouette", "Compares cohesion to separation; larger values often indicate clearer groups."),
        ("Davies-Bouldin", "Measures cluster similarity; lower is generally better."),
        ("Calinski-Harabasz", "Compares between-cluster to within-cluster dispersion; higher is generally better."),
        ("Choosing K and elbow", "An inertia bend is a clue, not proof of a unique correct K."),
        ("PCA visualization", "A 2D projection can help inspect structure but may hide higher-dimensional relationships."),
        ("Cluster profiling", "Compare feature distributions and sizes before proposing segment names.")],
     "Treating a PCA scatter as the original feature coordinates", "Which metric generally prefers lower values?", ["Davies-Bouldin", "Silhouette", "Calinski-Harabasz"], 0,
     "Compare K and profile each cluster before naming it.", "Why might a cluster visualization be misleading?"),
    ("advanced-depth", "Advanced ML decisions", "ADVANCED", "Advanced ML", "automl-explainability", [
        ("Feature engineering", "Create defensible predictors from training data without using future information."),
        ("Cross-validation", "Repeat training across folds for more stable estimates while keeping final test separate."),
        ("Hyperparameters", "Choose settings on validation evidence, not final test performance."),
        ("Regularization and ensembles", "Penalties constrain complexity; ensembles combine model strengths."),
        ("Calibration and threshold", "Check whether predicted probabilities match observed frequency; choose thresholds by decision costs."),
        ("Explainability", "Importance describes model dependence, not causal effects."),
        ("Fairness", "Compare error rates across relevant groups where law and context permit."),
        ("Drift", "Monitor shifts in input patterns and outcomes after deployment."),
        ("Deployment", "Version the fitted pipeline, validate input schema, and monitor predictions and failures.")],
     "Tuning repeatedly against the final test split", "What should stay separate from hyperparameter selection?", ["Final test", "Training", "Validation"], 0,
     "Write a monitoring checklist for a saved model.", "How would you detect drift or unfair error differences?")]

for key, title, level, stage, prerequisite, sections, mistake, question, choices, answer, activity, interview in ML_DEPTH:
    item = topic("automl", key, title, level, sections[0][1], "Use the AutoML results to inspect this decision.",
                 mistake, question, choices, answer, [heading for heading, _ in sections],
                 stage=stage, prerequisite=prerequisite, activity="practical")
    item.update(theory_sections=[{"heading": heading, "text": text} for heading, text in sections],
                practice_activity=activity, coding_exercise="Reproduce a small related workflow in Python Lab where relevant.",
                case_study="How would this choice affect a stroke-risk or customer-churn case?",
                interview_competencies=[interview])
    TOPICS.append(item)

TOPICS.extend([
    topic("automl", "metrics", "Model evaluation fundamentals", "BEGINNER", "Use held-out metrics that match the cost of mistakes. Accuracy alone can hide rare positive cases; MAE and RMSE describe numeric errors.", "For a rare stroke outcome, inspect recall and the confusion matrix alongside accuracy.", "A high accuracy score does not guarantee that positive cases were found.", "Which metric measures the fraction of actual positives detected?", ["Recall", "MAE", "Training time"], 0, ["accuracy", "recall", "evaluation"], stage="Evaluation", prerequisite="automl-splits"),
    topic("autodl", "training", "Activation, loss, and training", "BEGINNER", "An activation adds nonlinearity. A loss measures prediction error; gradients and an optimizer update weights. Epochs pass through training data in batches.", "A batch computes loss, backward gradients, then an optimizer step.", "Validation data should not update weights.", "What does the loss function measure?", ["Prediction error", "Number of classes", "Image width"], 0, ["activation", "loss", "backpropagation", "epochs", "batches"], prerequisite="autodl-neurons"),
    topic("autonlp", "preprocessing", "Text preparation and tokens", "FOUNDATION", "Tokenization splits text into units. Stopword removal, stemming, and lemmatization can lose useful context; choose them for the task.", "The phrase 'not good' may change meaning if 'not' is removed.", "Removing every common word can destroy negation.", "What does tokenization produce?", ["Text units", "Class probabilities", "Model weights"], 0, ["tokens", "stopwords", "stemming", "lemmatization"]),
])

# Additional, independent study units. Existing lesson IDs and capstone flows stay stable.
# Each row supplies its own check and illustration; none requires a saved lab artifact.
BALANCED_UNITS = {
    "autodl": [
        ("foundations", "What deep learning adds", "FOUNDATION", None, "Deep learning uses trainable layers to learn representations from data; it is one part of machine learning.", "An image model learns edges before combining them into objects.", "More layers do not guarantee better held-out results.", "What makes a neural network learn?", "Updating weights from training loss", "Changing test labels", "Adding layers without training", "python", "# Layers learn from examples through weight updates\nmodel = torch.nn.Linear(4, 2)"),
        ("tensors", "Tensors and shapes", "BEGINNER", "autodl-foundations", "A tensor is a numeric array. Image batches commonly use batch, channel, height and width dimensions.", "A batch of eight RGB 32x32 images has shape (8, 3, 32, 32).", "Mixing up batch and channel dimensions breaks model inputs.", "Which dimension is the batch size in (8, 3, 32, 32)?", "The first dimension: 8", "The second dimension: 3", "The last dimension: 32", "python", "images = torch.zeros(8, 3, 32, 32)\nprint(images.shape)"),
        ("layers", "Layers and networks", "BEGINNER", "autodl-tensors", "Input, hidden and output layers transform features step by step; a layer's output shape must match the next layer's input.", "A small network maps four measurements through eight hidden units to two class scores.", "Do not confuse an output score with a calibrated probability.", "What must connect consecutive layers?", "Compatible output and input shapes", "Identical layer names", "The final test labels", "python", "net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))"),
        ("activation", "Activation functions", "BEGINNER", "autodl-layers", "Activations such as ReLU add nonlinearity; sigmoid and softmax have different output interpretations.", "ReLU changes negative inputs to zero while retaining positive values.", "Stacking linear layers without nonlinear activation remains a linear mapping.", "Why place ReLU between linear layers?", "To add nonlinearity", "To split the dataset", "To save the artifact", "python", "x = torch.tensor([-2.0, 3.0])\nprint(torch.relu(x))  # [0, 3]"),
        ("forward", "Forward pass", "BEGINNER", "autodl-activation", "A forward pass sends inputs through layers to produce predictions before loss is calculated.", "An image batch becomes one score vector per image.", "A forward pass alone does not update weights.", "What does a forward pass produce?", "Model outputs from inputs", "Updated optimizer weights", "A held-out split", "python", "scores = net(torch.zeros(2, 4))\nprint(scores.shape)  # (2, 2)"),
        ("loss", "Loss functions", "BEGINNER", "autodl-forward", "A loss quantifies prediction error; choose a loss that matches the task and output representation.", "Cross-entropy compares class scores with known class labels.", "Accuracy is not a differentiable replacement for training loss.", "What does classification loss compare?", "Predictions with training labels", "Input width with height", "Only model size", "python", "loss_fn = torch.nn.CrossEntropyLoss()\nloss = loss_fn(scores, torch.tensor([0, 1]))"),
        ("backprop", "Backpropagation", "INTERMEDIATE", "autodl-loss", "Backward propagation computes gradients of loss with respect to trainable parameters.", "A gradient tells how a small weight change would affect current loss.", "Calling backward does not itself apply an optimizer step.", "What does loss.backward() compute?", "Parameter gradients", "Validation labels", "Saved model files", "python", "optimizer.zero_grad()\nloss.backward()  # gradients are now available"),
        ("optimizers", "Optimizers", "INTERMEDIATE", "autodl-backprop", "An optimizer uses gradients to update parameters. SGD and Adam use different update rules.", "After backward, step applies the update for this batch.", "Forgetting zero_grad can accumulate gradients unintentionally.", "Which call changes parameters?", "optimizer.step()", "loss.backward() alone", "model.eval()", "python", "optimizer = torch.optim.Adam(net.parameters(), lr=0.001)\noptimizer.zero_grad()\nloss.backward()\noptimizer.step()"),
        ("learning-rate", "Learning rate", "INTERMEDIATE", "autodl-optimizers", "Learning rate controls update size; values that are too high can overshoot a useful solution.", "Small steps may learn slowly; large steps may make loss unstable.", "Do not assume a larger learning rate always trains faster successfully.", "What can an excessive learning rate cause?", "Unstable or diverging loss", "Automatic generalization", "More input classes", "python", "optimizer = torch.optim.Adam(net.parameters(), lr=0.001)"),
        ("regularization", "Overfitting and regularization", "INTERMEDIATE", "autodl-training", "A widening training-validation gap can indicate overfitting. Dropout and suitable augmentation can help, but validation evidence matters.", "Training accuracy is 98% while validation accuracy is 73%.", "Do not infer generalization from training accuracy alone.", "What does a large training-validation gap suggest?", "Possible overfitting", "Guaranteed robust predictions", "Missing labels only", "python", "dropout = torch.nn.Dropout(0.5)\nmodel.train()  # dropout active\nmodel.eval()   # validate without dropout"),
        ("transfer", "Transfer learning", "ADVANCED", "autodl-cnn", "Transfer learning adapts an existing pretrained representation to a related task when data are limited.", "Reuse image features, then train a task-specific classifier head.", "A pretrained model is not automatically suitable for different imagery.", "Why try transfer learning with limited images?", "To reuse learned representations", "To avoid held-out evaluation", "To create labels automatically", "pseudocode", "pretrained image features -> new class head -> fine-tune -> validate"),
    ],
    "autonlp": [
        ("foundations", "Text as data", "FOUNDATION", None, "NLP turns text into representations a model can learn from while preserving the meaning needed for the task.", "A ticket message and its routing label form one supervised example.", "A raw sentence is not already a numeric feature vector.", "What does a text classifier learn from?", "Text examples paired with labels", "Labels without messages", "Only document length", "pseudocode", "ticket text -> representation -> classifier -> label"),
        ("cleaning", "Text cleaning", "BEGINNER", "autonlp-foundations", "Normalize obvious noise carefully; case and punctuation may carry meaning in some tasks.", "Trim surrounding spaces before vectorizing tickets.", "Removing all punctuation can erase sentiment or identifiers.", "What is a safe first cleaning step?", "Inspect noise before normalizing", "Delete all words", "Remove every symbol blindly", "python", "cleaned = text.strip()  # keep task-relevant punctuation"),
        ("stopwords", "Stopwords and negation", "BEGINNER", "autonlp-preprocessing", "Common words can be removed for some count models, but negation may reverse sentiment.", "Removing 'not' from 'not useful' changes the message.", "Do not apply a generic stopword list without checking the task.", "Why might 'not' be retained?", "It can reverse meaning", "It is always a label", "It creates a bigram by itself", "python", "tokens = ['not', 'useful']\nkept = [word for word in tokens if word not in {'the', 'a'}]"),
        ("word-forms", "Stemming and lemmatization", "BEGINNER", "autonlp-preprocessing", "Stemming trims word endings; lemmatization aims for dictionary forms. Both can lose nuance.", "'running' may become a shorter shared form with 'run'.", "A stem is not guaranteed to be a normal word.", "Which method aims at a dictionary form?", "Lemmatization", "Random truncation", "Label encoding", "pseudocode", "running -> run  # illustrative normalization; inspect meaning"),
        ("bow", "Bag of Words", "BEGINNER", "autonlp-preprocessing", "Bag of Words counts terms, making text numeric while ignoring much of word order.", "Two tickets can share word counts despite different order.", "Counts alone do not capture all context.", "What does Bag of Words mainly represent?", "Term counts", "Word order perfectly", "Class probabilities", "python", "from sklearn.feature_extraction.text import CountVectorizer\ncounts = CountVectorizer().fit_transform(['slow delivery', 'fast delivery'])"),
        ("ngrams", "N-grams", "INTERMEDIATE", "autonlp-tfidf", "An n-gram captures adjacent tokens; bigrams can retain short phrases such as 'not useful'.", "Unigrams see 'not' and 'useful' separately; a bigram sees their pairing.", "More n-grams can greatly enlarge the vocabulary.", "What does a bigram capture?", "Two adjacent tokens", "Only the class label", "The whole document as one token", "python", "from sklearn.feature_extraction.text import TfidfVectorizer\nvectorizer = TfidfVectorizer(ngram_range=(1, 2))"),
        ("pipeline", "Text classification pipeline", "INTERMEDIATE", "autonlp-tfidf", "Fit the vectorizer on training text, then transform held-out text before measuring a classifier.", "TF-IDF features feed a logistic regression classifier.", "Fitting vocabulary on test text leaks information.", "Where should vocabulary be fitted?", "Training text only", "Held-out text only", "All text before splitting", "python", "from sklearn.pipeline import make_pipeline\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\nmodel = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=200))"),
        ("embeddings", "Embeddings", "INTERMEDIATE", "autonlp-tfidf", "Embeddings are learned numeric vectors that can place semantically related text closer together.", "'car' and 'vehicle' can have nearby representations in a suitable model.", "A conceptual vector is not a trained NxZenAI embedding.", "What is an embedding?", "A learned numeric representation", "A class label", "A SQL table", "pseudocode", "'car' -> [0.2, -0.4, ...]  # conceptual vector"),
        ("context", "Context and sequence", "INTERMEDIATE", "autonlp-ngrams", "Meaning can depend on word order and surrounding text, which simple term counts only partly retain.", "'bank of a river' differs from 'bank approved a loan'.", "Matching a keyword alone need not resolve meaning.", "Why can context matter?", "The same word can mean different things", "Every word has one meaning", "Order never matters", "pseudocode", "bank + river context -> shoreline\nbank + loan context -> institution"),
        ("attention", "Attention", "ADVANCED", "autonlp-context", "Attention lets a model weight relationships between positions in a sequence; classroom highlights are illustrative, not real model weights.", "In a loan sentence, 'declined' may depend on 'credit score'.", "A conceptual attention diagram is not an explanation of one saved model.", "What does attention help model?", "Relationships among text positions", "Only total word count", "Database joins", "pseudocode", "token positions -> contextual relevance weights -> updated representations"),
        ("baseline", "Baseline versus transformer", "ADVANCED", "autonlp-transformers", "A simple TF-IDF classifier is a useful baseline. More complex models need measured held-out gains to justify cost.", "Compare both approaches on the same split and per-class metrics.", "Do not choose a transformer just because it is newer.", "When prefer the simple baseline?", "When it meets quality needs at lower cost", "Whenever test data are missing", "Only when labels are absent", "pseudocode", "same split + same metric -> compare TF-IDF baseline and transformer"),
    ],
    "python": [
        ("operators", "Operators and expressions", "BEGINNER", "python-basics", "Arithmetic calculates values; comparisons and Boolean operators build conditions.", "A spending threshold combines >= with and.", "Use == for comparison, not assignment.", "What does 3 * 2 evaluate to?", "6", "5", "32", "python", "total = 3 * 2\nallowed = total >= 5 and total < 10"),
        ("conditions", "Conditions", "BEGINNER", "python-basics", "if, elif and else select one branch based on Boolean conditions.", "A discount rule checks customer status before applying a rate.", "An else branch does not need another condition.", "Which branch runs when the first if condition is false but elif is true?", "The elif branch", "The if branch", "Every branch", "python", "if amount > 100:\n    tier = 'high'\nelif amount > 0:\n    tier = 'standard'\nelse:\n    tier = 'none'"),
        ("loops", "Loops and comprehensions", "BEGINNER", "python-conditions", "for iterates over items; a list comprehension creates a new list from an iterable.", "Collect positive amounts without changing the original list.", "Avoid changing a list while iterating through it.", "What does [x for x in [1, -2, 3] if x > 0] return?", "[1, 3]", "[-2]", "[1, -2, 3]", "python", "positive = [value for value in amounts if value > 0]"),
        ("maps-sets", "Dictionaries and sets", "BEGINNER", "python-collections", "Dictionaries map keys to values; sets keep unique members and support membership checks.", "A customer record maps 'city' to its value; a set finds distinct cities.", "A missing dictionary key can raise KeyError.", "What does len(set(['A', 'A', 'B'])) return?", "2", "3", "1", "python", "customer = {'city': 'Chennai'}\ncities = {'Chennai', 'Mumbai'}\nprint(customer.get('city'), len(cities))"),
        ("parameters", "Parameters and scope", "INTERMEDIATE", "python-collections", "Parameters receive arguments; return passes a value to the caller. Local names do not automatically change outer names.", "A function accepts a tax rate with a safe default.", "Avoid a mutable list or dictionary as a default argument.", "What does return do?", "Sends a result to the caller", "Prints every local variable", "Changes every global name", "python", "def total_with_tax(amount, rate=0.05):\n    return amount * (1 + rate)"),
        ("numpy", "NumPy arrays and shapes", "INTERMEDIATE", "python-collections", "NumPy arrays support numeric shapes and elementwise operations; inspect shape before combining data.", "A two-row, three-column matrix has shape (2, 3).", "Array multiplication is elementwise, not matrix multiplication.", "What does an array shape of (2, 3) describe?", "Two rows and three columns", "Three rows and two columns", "Two labels and three files", "python", "import numpy as np\nvalues = np.array([[1, 2, 3], [4, 5, 6]])\nprint(values.shape)"),
        ("filter-group", "Filtering and aggregating data", "INTERMEDIATE", "python-pandas", "Filter rows with a Boolean mask; groupby plus an aggregation summarizes groups.", "Compare total sales for each region after excluding canceled rows.", "A groupby object alone does not produce totals.", "Which expression sums sales per region?", "df.groupby('region')['amount'].sum()", "df.head()", "df['region'].unique()", "python", "active = df.loc[df['status'] != 'cancelled']\ntotals = active.groupby('region')['amount'].sum()"),
        ("plotting", "Matplotlib basics", "INTERMEDIATE", "python-pandas", "Choose axes and labels before plotting; use a line for ordered values and bars for categories.", "Plot monthly revenue along a meaningful time axis.", "A line between unrelated categories implies an order that may not exist.", "Which chart suits monthly revenue over time?", "Line chart", "Confusion matrix", "Unordered pie slices only", "python", "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot(months, revenue)\nax.set(xlabel='Month', ylabel='Revenue')"),
        ("reusable", "Reusable data functions", "ADVANCED", "python-pandas", "Small pure functions make repeated cleaning and analysis easier to review and reuse.", "A helper receives a DataFrame and returns a filtered copy.", "Do not silently mutate a caller's DataFrame when a copy is expected.", "What helps a data function be reusable?", "Clear inputs and returned output", "Hidden global state", "Hardcoded local file paths", "python", "def positive_sales(frame):\n    return frame.loc[frame['amount'] > 0].copy()"),
    ],
    "sql": [
        ("relations", "Relational tables and keys", "FOUNDATION", None, "Tables hold rows and columns; keys relate records across tables.", "orders.customer_id identifies a matching customers.id.", "Joining on similarly named but unrelated columns can multiply rows.", "What links an order to its customer?", "A matching key", "The row position", "A chart label", "sql", "SELECT o.customer_id FROM orders AS o;"),
        ("sort-distinct", "ORDER BY and DISTINCT", "BEGINNER", "sql-select", "ORDER BY sorts output; DISTINCT removes duplicate result rows.", "List each city once in alphabetical order.", "DISTINCT applies to the selected combination of columns.", "Which clause sorts results?", "ORDER BY", "WHERE", "HAVING", "sql", "SELECT DISTINCT city FROM customers ORDER BY city;"),
        ("aggregates", "Aggregate functions", "BEGINNER", "sql-select", "COUNT, SUM and AVG summarize rows; check whether NULL values are ignored by a given aggregate.", "Count orders and sum quantities.", "COUNT(column) and COUNT(*) can differ when values are NULL.", "Which function adds numeric values?", "SUM", "ORDER BY", "DISTINCT", "sql", "SELECT COUNT(*) AS order_count, SUM(quantity) AS units FROM orders;"),
        ("left-join", "LEFT JOIN", "INTERMEDIATE", "sql-joins", "LEFT JOIN retains unmatched left rows and fills missing right values with NULL.", "Keep customers who placed no orders.", "A right-table filter in WHERE can accidentally drop unmatched rows.", "Which join keeps all customers?", "LEFT JOIN customers to orders", "INNER JOIN only", "CROSS JOIN only", "sql", "SELECT c.name, o.quantity FROM customers c LEFT JOIN orders o ON o.customer_id = c.id;"),
        ("subqueries", "Subqueries", "INTERMEDIATE", "sql-groups", "A subquery computes a value or set used by an outer SELECT.", "Find products priced above the catalog average.", "A subquery can return multiple rows when a scalar comparison expects one.", "What can a scalar subquery provide?", "One value for an outer condition", "Automatic indexing", "A new database", "sql", "SELECT id FROM products WHERE price > (SELECT AVG(price) FROM products);"),
        ("ctes", "Common table expressions", "INTERMEDIATE", "sql-joins", "A CTE names an intermediate SELECT. It helps read complex queries but the fixed Learning SQL grader currently accepts SELECT-starting submissions only.", "Explore WITH queries in SQL Lab; submit a SELECT-based solution for the fixed challenge.", "Do not paste a WITH-starting illustration into the current challenge grader.", "What does a CTE name?", "An intermediate query result", "A model artifact", "A chart axis", "sql", "WITH totals AS (SELECT customer_id, SUM(quantity) AS units FROM orders GROUP BY customer_id) SELECT * FROM totals;  -- SQL Lab illustration, not a grader submission"),
        ("case-null", "CASE and NULL handling", "INTERMEDIATE", "sql-groups", "CASE derives conditional values; IS NULL checks missing values, and COALESCE supplies a fallback.", "Label missing cities as Unknown for a report.", "NULL is not equal to NULL using =.", "Which expression checks missing values?", "IS NULL", "= NULL", "ORDER NULL", "sql", "SELECT name, CASE WHEN city IS NULL THEN 'Unknown' ELSE city END AS city_label FROM customers;"),
        ("business", "Business analysis query", "ADVANCED", "sql-revenue", "A business query joins on known keys, aggregates the requested measure and checks whether joins inflate totals.", "Report customer revenue and order-line counts together, then check whether joins inflated either value.", "Summing prices without quantity understates revenue.", "How is revenue calculated in the fixed fixture?", "SUM(price * quantity) by customer", "COUNT(city) by region", "MAX(customer_id) only", "sql", "SELECT c.name, COUNT(*) AS order_lines, SUM(p.price * o.quantity) AS total_revenue FROM customers c JOIN orders o ON o.customer_id = c.id JOIN products p ON p.id = o.product_id GROUP BY c.name;"),
    ],
    "eda": [
        ("purpose", "Why explore data?", "FOUNDATION", None, "EDA describes data quality and patterns before conclusions or modeling.", "Inspect shape and missingness before choosing a chart.", "A polished chart cannot repair incorrect source data.", "What should come before interpretation?", "Inspect data quality", "Assume missing values are zero", "Train a model immediately", "python", "print(df.shape)\nprint(df.dtypes)"),
        ("types", "Types and units", "BEGINNER", "eda-schema", "Column types and units determine sensible summaries and visualizations.", "Age in years and income in currency need different interpretation.", "A numeric ID is not a continuous measurement.", "What should be checked before averaging a column?", "Meaning, type and unit", "Only its name length", "Only row order", "python", "print(df.dtypes)\nprint(df[['age', 'income']].describe())"),
        ("categories", "Categorical distributions", "BEGINNER", "eda-schema", "Value counts reveal common and rare categories; include missing categories deliberately.", "Count how many records belong to each department.", "Percentages can hide a very small group size.", "Which method counts categories?", "value_counts()", "corr()", "plot.scatter()", "python", "print(df['department'].value_counts(dropna=False))"),
        ("outliers", "Outliers and skew", "INTERMEDIATE", "eda-quality", "Extreme values may be errors or genuine cases; inspect them and the distribution before changing data.", "A salary 100 times the median deserves investigation.", "Do not delete every extreme value automatically.", "What should happen before removing an outlier?", "Inspect its source and distribution", "Delete the row immediately", "Replace all salaries with zero", "python", "q1, q3 = df['salary'].quantile([0.25, 0.75])\niqr = q3 - q1\nprint(q1 - 1.5 * iqr, q3 + 1.5 * iqr)"),
        ("transformations", "Transformations and leakage", "ADVANCED", "eda-quality", "Record cleaning changes and keep final evaluation data separate when a transformation is learned from data.", "Compare missing counts before and after a documented fill rule.", "Fitting a scaler on held-out rows leaks evaluation information.", "Where should a learned transform be fitted?", "Training rows only", "Final test rows first", "Every row before splitting", "python", "print(df.isna().sum())\n# Document changes before saving a cleaned copy"),
    ],
    "genai": [
        ("tokens", "Tokens and context windows", "BEGINNER", "genai-llm", "Models read and generate tokens; the context window limits what can fit in one request.", "A long document may require selecting relevant passages.", "A context window is not a guarantee the model used every sentence.", "What limits text available in one model request?", "Context window", "Chart width", "Database row count", "pseudocode", "input tokens + output tokens <= model context budget"),
        ("temperature", "Temperature and repeatability", "BEGINNER", "genai-llm", "Temperature changes sampling variability; lower values often produce steadier output but not guaranteed truth.", "Use a lower setting for repeatable extraction, then verify the content.", "Low temperature does not prevent hallucination.", "What does lower temperature usually reduce?", "Sampling variability", "Need for evidence", "Context size", "json", '{"temperature": 0.2, "note": "illustrative setting"}'),
        ("hallucinations", "Hallucinations and uncertainty", "BEGINNER", "genai-llm", "A fluent answer can include unsupported claims. Ask for uncertainty and verify against reliable sources.", "If the selected documents lack an answer, say so.", "Do not treat confident prose as evidence.", "What should happen when a source lacks an answer?", "Say evidence is insufficient", "Invent a citation", "Claim certainty", "prompt", "Answer only from the supplied context. If evidence is missing, say so."),
        ("embeddings", "Embeddings and vector search", "INTERMEDIATE", "genai-tokens", "Embeddings map text to vectors so search can retrieve passages with related meaning.", "A question about refunds may retrieve a returns-policy paragraph.", "Vector similarity alone does not prove factual support.", "What does vector search help find?", "Semantically related passages", "Verified truth automatically", "New training labels", "pseudocode", "question -> embedding -> similar passages -> inspect relevance"),
        ("retrieval", "Chunking and retrieval", "INTERMEDIATE", "genai-embeddings", "Long documents are split into passages; retrieval selects likely relevant chunks for a question.", "Retrieve the paragraph about a policy rather than the entire handbook.", "Poor chunk boundaries can omit qualifying context.", "Why chunk a long document?", "To retrieve relevant passages within context limits", "To guarantee correctness", "To remove all citations", "pseudocode", "document -> chunks -> index -> retrieve relevant chunks"),
        ("citations", "Grounding and citations", "INTERMEDIATE", "genai-rag", "A grounded response ties claims to selected source passages and admits missing evidence.", "Cite the policy section supporting each answer claim.", "A citation to an unrelated document does not support a claim.", "What makes a citation useful?", "It supports the specific claim", "It is long", "It looks official", "prompt", "Use selected documents only. Cite supporting passages; state when support is absent."),
        ("evaluation", "Evaluating GenAI responses", "ADVANCED", "genai-rag", "Compare answers against the task, source support, completeness and safe failure behavior.", "Check whether a cited answer actually addresses the question.", "A longer response need not be more accurate.", "Which criterion tests grounding?", "Claims are supported by cited evidence", "Response length alone", "Model brand alone", "pseudocode", "task + evidence + rubric -> check answer support and completeness"),
        ("guardrails", "Tool safety and failure", "ADVANCED", "genai-agents", "Tool use needs complete arguments, permission boundaries and a safe response when a tool fails.", "A tool failure should be reported rather than replaced with invented data.", "Do not expose secret credentials in tool arguments.", "What should an agent do when a required tool fails?", "Explain the failure without inventing results", "Invent a successful result", "Retry forever", "pseudocode", "goal -> select tool -> validate arguments -> call -> inspect outcome -> answer or fail safely"),
    ],
    "prompting": [
        ("context", "Context and audience", "BEGINNER", "prompting-clarity", "A prompt states who the answer is for and what background the model should use.", "Explain overfitting to first-year students with one everyday analogy.", "An unstated audience produces inconsistent depth.", "What helps tailor an explanation?", "Audience and relevant background", "Only a model name", "More adjectives without context", "prompt", "Explain overfitting to first-year engineering students using one everyday analogy."),
        ("format", "Constraints and output format", "BEGINNER", "prompting-clarity", "State measurable limits and the desired structure so output can be checked.", "Ask for three numbered steps and a short summary.", "An unspecified format is hard to compare.", "Which instruction is testable?", "Return exactly three numbered steps", "Be excellent", "Write something useful", "prompt", "Describe the process in exactly three numbered steps, then one sentence of summary."),
        ("variables", "Prompt variables", "BEGINNER", "prompting-clarity", "Variables let one template safely accept different user inputs without rewriting its instructions.", "The same explanation prompt can accept a different topic each run.", "An unresolved placeholder can confuse the model.", "What should happen before running a template?", "Resolve every required variable", "Leave placeholders in place", "Delete the task", "prompt", "Explain {{topic}} to {{audience}} in three bullets."),
        ("few-shot", "Role and few-shot examples", "INTERMEDIATE", "prompting-context", "A useful role clarifies perspective; a few consistent examples show the desired input-output pattern.", "Two labeled sentiment examples guide a third classification.", "Contradictory examples undermine the instruction.", "What makes few-shot examples useful?", "Consistent input-output patterns", "Unrelated examples", "Hidden grading rules", "prompt", "You are a support triage assistant.\nInput: Cannot sign in -> Output: Account\nInput: Charged twice -> Output: Billing\nInput: Reset my password -> Output:"),
        ("extraction", "Structured extraction", "INTERMEDIATE", "prompting-format", "Specify fields, allowed values and missing-value behavior for information extraction.", "Extract name and skills from a resume into a JSON object.", "A JSON request without a schema may omit fields.", "What improves structured extraction?", "Explicit fields and missing-value rules", "Only a role", "No output schema", "prompt", "Extract candidate_name and skills from the supplied text. Return only JSON; use null for missing name and [] for missing skills."),
        ("reduce-hallucination", "Handling unsupported claims", "INTERMEDIATE", "prompting-grounding", "Restrict claims to supplied evidence and state what remains unknown.", "A missing policy answer should produce an insufficiency message.", "A source-only instruction without documents is not grounding.", "What should a prompt request for missing evidence?", "Explicit uncertainty or refusal", "An invented source", "A confident guess", "prompt", "Use only selected documents. Cite each supported claim. If none supports the answer, state that evidence is insufficient."),
        ("evaluation", "Prompt evaluation and comparison", "ADVANCED", "prompting-compare", "Use the same task and rubric across runs to compare instruction following, evidence and formatting.", "Compare two saved outputs against identical source documents.", "Different prompts with different tasks cannot isolate prompt quality.", "What makes a prompt comparison fair?", "Shared task and scoring criteria", "Only response length", "Different source sets", "prompt", "Evaluate each answer for source support, completeness and required format using the same rubric."),
        ("tool-agent", "Tool-use and agent prompts", "ADVANCED", "prompting-grounding", "A tool prompt says when to call a tool, what arguments it needs and when to stop; it does not grant permissions.", "Use a database tool only when a current record is needed.", "Do not instruct a tool to run without required arguments.", "What belongs in a tool-use prompt?", "Trigger, arguments and failure behavior", "Hidden credentials", "A fabricated tool result", "prompt", "If current order status is needed, request the order lookup with a validated order ID. If no ID is supplied, ask for it; never invent status."),
    ],
}

for module, units in BALANCED_UNITS.items():
    for (key, title, level, prerequisite, theory, example, mistake, question,
         correct, wrong_a, wrong_b, example_type, snippet) in units:
        item = topic(module, key, title, level, theory, example, mistake, question,
                     [correct, wrong_a, wrong_b], 0, [key.replace('-', ' '), title.lower()],
                     stage="Advanced" if level == "ADVANCED" else "Core Skills",
                     prerequisite=prerequisite)
        item["illustrative_example"] = {"type": example_type, "title": title, "content": snippet,
                                         "note": "Illustrative learning example."}
        TOPICS.append(item)

CAPSTONES = {
    "automl-classification-capstone": ("Machine Learning Classification Business Case", "classification", ["automl-features-target", "automl-splits", "automl-preparation", "automl-classification-depth", "automl-metrics"]),
    "automl-regression-capstone": ("Machine Learning Regression Business Case", "regression", ["automl-features-target", "automl-splits", "automl-preparation", "automl-regression-depth", "automl-metrics"]),
    "automl-clustering-capstone": ("Machine Learning Clustering Business Case", "clustering", ["automl-features-target", "automl-splits", "automl-preparation", "automl-clustering-depth"]),
    "autodl-image-capstone": ("Deep Learning Image Business Case", "image_classification", ["autodl-neurons", "autodl-training", "autodl-cnn", "autodl-pytorch"]),
    "autonlp-text-capstone": ("NLP Text Business Case", "text_classification", ["autonlp-preprocessing", "autonlp-tfidf", "autonlp-metrics", "autonlp-transformers"]),
}

# Shared, deterministic capstone workflow. Stage answers live in learning_progress;
# only the final saved-result check awards verified practical evidence.
LIFECYCLE_STAGE_IDS = (
    "BUSINESS_UNDERSTANDING", "PROBLEM_FORMULATION", "DATA_UNDERSTANDING", "EDA",
    "DATA_PREPARATION", "BASELINE", "MODEL_DEVELOPMENT", "EVALUATION",
    "ERROR_ANALYSIS", "PREDICTION", "INTERPRETATION", "BUSINESS_RECOMMENDATION",
    "VERIFICATION",
)
LIFECYCLE_TITLES = {
    "BUSINESS_UNDERSTANDING": "Business problem", "PROBLEM_FORMULATION": "Problem formulation",
    "DATA_UNDERSTANDING": "Data understanding", "EDA": "Data quality / EDA",
    "DATA_PREPARATION": "Data preparation", "BASELINE": "Baseline",
    "MODEL_DEVELOPMENT": "Model development", "EVALUATION": "Evaluation",
    "ERROR_ANALYSIS": "Error analysis", "PREDICTION": "Prediction / inference",
    "INTERPRETATION": "Learner interpretation", "BUSINESS_RECOMMENDATION": "Business recommendation",
    "VERIFICATION": "Capstone verification",
}
LIFECYCLE_PROBLEMS = {
    "automl-classification-capstone": ("Stroke risk: use labeled patient records to estimate which cases need closer review.", "A clinician or care coordinator reviewing positive and missed cases.", "Missed positives can delay care; false alarms can increase unnecessary review."),
    "automl-regression-capstone": ("House prices: estimate sale price from property features to support planning.", "A buyer, seller or analyst comparing estimates with actual prices.", "Large errors can lead to poor pricing or budget decisions."),
    "automl-clustering-capstone": ("Customer segmentation: group similar customers without a target label.", "A business team planning different actions for each segment.", "Misleading segments can waste effort or hide important groups."),
    "autodl-image-capstone": ("Defect classification: identify normal and defective items from labeled images.", "A quality reviewer checking model-assisted decisions.", "Missed defects and false alarms both carry operational costs."),
    "autonlp-text-capstone": ("Support ticket classification: categorize incoming text so tickets reach the right team.", "A support operations team reviewing routed tickets.", "Misclassified tickets can delay or misdirect help."),
}
LIFECYCLE_INSTRUCTIONS = {
    "BUSINESS_UNDERSTANDING": "Read the business problem and describe the decision this model could support.",
    "PROBLEM_FORMULATION": "Choose the supported task and primary metric; record why they fit the decision.",
    "DATA_UNDERSTANDING": "Inspect the dataset in the lab. Record its size, input fields and target or labels where applicable.",
    "EDA": "Inspect quality in the existing lab. AutoML may also open EDA Hub; no cross-lab transfer is assumed.",
    "DATA_PREPARATION": "Review the preparation shown by the lab and record what was applied or what remains uncertain.",
    "BASELINE": "Compare an existing simple candidate if available; otherwise record a conceptual baseline. No extra training is started here.",
    "MODEL_DEVELOPMENT": "Train in the existing lab, then select a compatible saved result created after this capstone started.",
    "EVALUATION": "Read the saved result in the lab and record the observed primary metric and comparison with the baseline.",
    "ERROR_ANALYSIS": "Inspect available confusion, residual, cluster, validation-gap or class metrics; describe a likely weakness.",
    "PREDICTION": "Try the lab's existing prediction flow if supported. Record the observed outcome; unsupported models may skip this stage.",
    "INTERPRETATION": "Explain the most important result, limitation and whether the model is suitable for this case.",
    "BUSINESS_RECOMMENDATION": "Choose deploy, improve or not ready, and justify the decision. This does not deploy a model.",
    "VERIFICATION": "Verify the selected owned result after completing the required stages.",
}
for capstone_id, (title, expected_task, prerequisites) in CAPSTONES.items():
    module, key = capstone_id.split("-", 1)
    item = topic(module, key, title, "CAPSTONE", "Apply mastered foundations in the existing lab, then verify a newly completed owned result.",
                 "Start this capstone before training a new model.", "A saved model proves training, not mastery of each prerequisite concept.",
                 "What should you do before verifying a capstone result?", ["Complete prerequisites and train for this activity", "Reuse any old model", "Skip evaluation"], 0,
                 ["prerequisites", "training", "verification"], stage="Capstone")
    item.update(is_capstone=True, expected_task=expected_task, prerequisites=prerequisites,
                activity_types=["CONCEPT", "PRACTICAL", "CASE"])
    if module == "autonlp":
        item["compatible_tasks"] = ["text_classification", "sentiment_analysis", "intent_classification"]
    TOPICS.append(item)

# Each answer key stays on the server. Explanations are recorded but are not auto-graded.
FOUNDATION_REASONING = {
    "automl-features-target": ("A bank table has customer_id, age, salary, credit_score, loan_amount, default.", "Which target and identifier choice is sound?", ["Predict default; exclude customer_id", "Predict customer_id; exclude default", "Predict salary; keep every column"], 0),
    "automl-splits": ("A team evaluates on the same records used for fitting.", "Why is this result unreliable?", ["The model has already seen those outcomes", "Training rows are always harder", "Test rows need no target"], 0),
    "automl-preparation": ("A churn table has customer_id, account_closed, churn_reason, monthly_spend, churn.", "Which fields are most likely to leak or memorize the outcome?", ["account_closed, churn_reason, and customer_id", "monthly_spend only", "churn only"], 0),
    "automl-imbalance": ("A medical classifier reports 95% accuracy and 32% recall.", "What should be investigated first?", ["Missed positive cases and the confusion matrix", "Only training speed", "Accuracy is sufficient"], 0),
    "automl-metrics": ("Only 2% of transactions are fraudulent.", "Which evaluation view helps reveal missed fraud?", ["Positive-class recall and precision–recall", "Accuracy alone", "Training duration"], 0),
    "automl-foundations-depth": ("A team predicts loan default from previously labeled customers and groups unlabeled customers.", "Which description is correct?", ["Default prediction is supervised; grouping is unsupervised", "Both are supervised", "Neither uses data"], 0),
    "automl-preparation-depth": ("A scaler is fitted before splitting the dataset.", "What is the safest correction?", ["Fit it on training rows only", "Fit it on test rows only", "Remove the test split"], 0),
    "automl-regression-errors": ("Two price models have equal MAE but different RMSE.", "What may explain the higher RMSE?", ["A few larger errors", "More class labels", "A perfect model"], 0),
    "automl-clustering": ("A team selects K only from the prettiest 2D plot.", "What else should they inspect?", ["Cluster profiles and separation metrics", "Only chart color", "Classification recall"], 0),
    "automl-explainability": ("Age is the most important feature in a fitted model.", "What can be concluded?", ["The model relies on age; causation is unproven", "Age causes the outcome", "The model is fair"], 0),
    "automl-advanced-depth": ("A team tunes on the final test split repeatedly.", "What is the concern?", ["The final estimate becomes biased by selection", "Training speeds up", "Features become numeric"], 0),
    "automl-classification-depth": ("A fraud classifier is highly accurate but misses most fraud.", "Which diagnostic informs the next decision?", ["Confusion matrix and positive-class recall", "Training accuracy alone", "Model size alone"], 0),
    "automl-regression-depth": ("A price model has low MAE but much higher RMSE.", "What is a plausible explanation?", ["Some large prediction errors", "Perfect calibration", "Too many class labels"], 0),
    "automl-clustering-depth": ("A 2D PCA plot shows three separated groups.", "What should be checked before naming them?", ["Original-feature profiles and sizes", "Only chart colors", "Classification accuracy"], 0),
    "autodl-neurons": ("A neuron computes an output from inputs and trainable parameters.", "What changes during learning?", ["Weights and bias through gradients", "Only the test labels", "The input image dimensions"], 0),
    "autodl-training": ("Training loss falls while validation loss rises.", "What does this suggest?", ["Possible overfitting", "Guaranteed generalization", "No gradients"], 0),
    "autodl-cnn": ("A CNN scores well on its training images.", "What checks generalization?", ["Separate held-out images", "Repeating the same images", "More training epochs only"], 0),
    "autodl-pytorch": ("A loop calls backward repeatedly without zero_grad.", "What happens?", ["Gradients accumulate", "Validation starts automatically", "The model is saved"], 0),
    "autonlp-preprocessing": ("A sentiment pipeline removes every stopword, including 'not'.", "What is the risk?", ["Negation meaning can be lost", "Vocabulary always improves", "Labels are removed"], 0),
    "autonlp-tfidf": ("The word 'the' appears in nearly every ticket.", "How does TF-IDF usually treat it?", ["Lower relative weight", "Guaranteed highest weight", "Turns it into a label"], 0),
    "autonlp-metrics": ("A ticket classifier rarely predicts a minority category.", "What should you inspect?", ["Per-class recall and confusion matrix", "Overall accuracy only", "Token count only"], 0),
    "autonlp-transformers": ("A small ticket dataset has a working TF-IDF baseline.", "When is a transformer justified?", ["When held-out gains justify data and compute cost", "Always", "Before measuring a baseline"], 0),
}
CODE_READING = {
    "automl-features-target": ("Which code separates inputs from target?", ['X = df.drop(columns=["target"]); y = df["target"]', 'X = df["target"]; y = df.drop(columns=["target"])', 'X = y = df'], 0),
    "automl-splits": ("Which call creates held-out rows?", ["train_test_split(X, y, test_size=0.2, random_state=42)", "model.fit(X, y); model.score(X, y)", "X.copy()"], 0),
    "automl-preparation": ("Which operation fills missing values using training-fitted values?", ["imputer.fit(X_train); imputer.transform(X_test)", "imputer.fit(X_test); imputer.transform(X_train)", "X_test.dropna() only"], 0),
    "automl-metrics": ("Which code measures positive-class recall?", ["recall_score(y_test, y_pred)", "mean_absolute_error(y_test, y_pred)", "len(X_train)"], 0),
    "autodl-pytorch": ("Which order performs one optimizer update?", ["zero_grad, forward, loss, backward, step", "step, forward, backward", "eval, step, loss"], 0),
    "autonlp-tfidf": ("Which operation fits vocabulary only on training text?", ["vectorizer.fit_transform(train_text)", "vectorizer.fit_transform(test_text)", "vectorizer.fit_transform(all_text)"], 0),
}
for item in TOPICS:
    if item["id"] in FOUNDATION_REASONING:
        scenario, question, choices, answer = FOUNDATION_REASONING[item["id"]]
        item["reasoning_check"] = {"scenario": scenario, "question": question, "choices": choices, "answer_index": answer}
        item["quiz_check"] = dict(item["quick_check"])
        concept_choices = ["Training accuracy alone proves this concept was applied correctly.",
                           "Any saved model proves the learner understands this concept."]
        concept_answer = sum(map(ord, item["id"])) % 3
        concept_choices.insert(concept_answer, item["description"])
        item["quick_check"] = {"question": f"Which statement correctly explains {item['title']}?",
                               "choices": concept_choices, "answer_index": concept_answer}
        item["activity_types"] = ["CONCEPT", "REASONING", "QUIZ", *(["CODING"] if item["id"] in CODE_READING else [])]
        if item["id"] in CODE_READING:
            question, choices, answer = CODE_READING[item["id"]]
            item["code_check"] = {"question": question, "choices": choices, "answer_index": answer}
    elif item["module"] in {"automl", "autodl", "autonlp"} and not item.get("is_capstone"):
        item["activity_types"] = ["CONCEPT", "QUIZ"]

# Application checks are separate from recall questions. Explanations are recorded,
# while the selected answer supplies the deterministic pass/fail result.
BALANCED_REASONING = {
    "autodl-regularization": ("Training accuracy is 98% but validation accuracy is 73%.", "What should be investigated?", "Possible overfitting and data mismatch", "Increase reported training accuracy only", "Ignore the validation set"),
    "autodl-transfer": ("Only a small labeled image collection is available.", "What is a reasonable first comparison?", "Try a pretrained feature extractor and validate it", "Assume a new large network will generalize", "Skip held-out images"),
    "autonlp-stopwords": ("A sentiment sentence says 'not useful'.", "What happens if 'not' is removed?", "The sentiment meaning may reverse", "Nothing can change", "The label becomes numeric"),
    "autonlp-pipeline": ("A vectorizer is fitted before a train/test split.", "What is the risk?", "Held-out vocabulary leaks into training", "The split becomes stratified", "Every word is translated"),
    "autonlp-baseline": ("A TF-IDF baseline is fast and meets the quality target.", "What should justify a transformer?", "A measured held-out gain worth its cost", "Its newer architecture alone", "A longer model name"),
    "python-reusable": ("A helper silently modifies the caller's data.", "How can its contract be clearer?", "Return a transformed copy and document inputs", "Use hidden global variables", "Mutate every input without notice"),
    "sql-left-join": ("A report must include customers with no orders.", "Which join is appropriate?", "LEFT JOIN from customers", "INNER JOIN only", "Filter for orders before joining"),
    "sql-business": ("A join unexpectedly doubles revenue totals.", "What should be checked first?", "Join keys and row cardinality", "Chart color", "Rename the total column only"),
    "eda-outliers": ("A salary is 100 times the median.", "What is the sound first action?", "Inspect the record and distribution", "Delete it immediately", "Assume all salaries are wrong"),
    "genai-hallucinations": ("Selected documents do not answer a question.", "What should the assistant do?", "State that evidence is insufficient", "Invent a plausible citation", "Present memory as document evidence"),
    "genai-guardrails": ("A tool request lacks a required record ID.", "What should the agent do?", "Ask for the ID before using the tool", "Invent an ID", "Call every available tool"),
    "prompting-evaluation": ("Two outputs used different tasks and source sets.", "How should they be compared fairly?", "Rerun with the same task, sources and rubric", "Pick the longer output", "Ignore source support"),
}
for item in TOPICS:
    if item["id"] in BALANCED_REASONING:
        scenario, question, correct, wrong_a, wrong_b = BALANCED_REASONING[item["id"]]
        item["reasoning_check"] = {"scenario": scenario, "question": question,
                                   "choices": [correct, wrong_a, wrong_b], "answer_index": 0}
        item["quiz_check"] = dict(item["quick_check"])
        item["quick_check"] = {"question": f"Which statement explains {item['title']}?",
                               "choices": [item["description"], "Any saved model proves this concept.",
                                           "The final test data should train the model."], "answer_index": 0}
        item["activity_types"] = ["CONCEPT", "REASONING", "QUIZ"]

BALANCED_CODE_READING = {
    "autodl-activation": ("What does torch.relu(torch.tensor([-2., 3.])) return?", ["[0., 3.]", "[-2., 3.]", "[2., -3.]"]),
    "autodl-optimizers": ("Which line applies the computed gradient update?", ["optimizer.step()", "loss.backward()", "model.eval()"]),
    "autodl-cnn": ("Which layer uses a moving convolution filter?", ["torch.nn.Conv2d(3, 16, 3)", "torch.nn.Linear(3, 16)", "torch.nn.Dropout(0.5)"]),
    "autodl-regularization": ("Which layer randomly drops training activations?", ["torch.nn.Dropout(0.5)", "torch.nn.Conv2d(3, 16, 3)", "torch.nn.Linear(3, 16)"]),
    "autonlp-pipeline": ("Where should a text vectorizer fit its vocabulary?", ["On training text", "On held-out text", "On all data before splitting"]),
    "python-loops": ("What does [x for x in [1, -2, 3] if x > 0] return?", ["[1, 3]", "[-2]", "[1, -2, 3]"]),
    "sql-left-join": ("Which query keeps customers without orders?", ["SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id", "SELECT c.name FROM customers c INNER JOIN orders o ON o.customer_id = c.id", "SELECT name FROM orders"]),
    "eda-categories": ("Which operation includes missing categories in counts?", ["df['department'].value_counts(dropna=False)", "df['department'].corr()", "df.head()"]),
}
for item in TOPICS:
    if item["id"] in BALANCED_CODE_READING:
        question, choices = BALANCED_CODE_READING[item["id"]]
        item["code_check"] = {"question": question, "choices": choices, "answer_index": 0}
        item["activity_types"] = list(dict.fromkeys([*item.get("activity_types", ["CONCEPT"]), "CODING"]))

BALANCED_VISUALS = {
    "autodl-layers": ("Passing decisions through a team", "Each layer refines the representation passed to the next.", "NEURAL_NETWORK"),
    "autodl-learning-rate": ("Step size downhill", "Large steps may overshoot; small steps may move slowly.", "GRADIENT_DESCENT"),
    "autodl-regularization": ("Practicing versus unseen questions", "Memorizing training examples can fail on new examples.", "NEURAL_NETWORK"),
    "autonlp-foundations": ("Turning words into signals", "Text must become a useful representation before prediction.", "PROCESS_FLOW"),
    "autonlp-attention": ("Focused readers", "Some words are more relevant to interpreting another word.", "ATTENTION"),
    "genai-retrieval": ("Finding the right page", "Retrieval narrows a large collection to relevant passages.", "PROCESS_FLOW"),
    "prompting-evaluation": ("Same exam for both drafts", "Use one task and one rubric for a fair comparison.", "PROCESS_FLOW"),
}
for item in TOPICS:
    if item["id"] in BALANCED_VISUALS:
        title, description, visual = BALANCED_VISUALS[item["id"]]
        item["analogy"] = {"title": title, "description": description}
        item["visual"] = {"type": visual, "config": {}}

# Present prerequisites before their dependents without changing stable topic IDs.
_topic_index = {item["id"]: item for item in TOPICS}
if len(_topic_index) != len(TOPICS):
    raise ValueError("Learning topic IDs must be unique.")
_ordered_topics = []
_visited_topics = set()
_visiting_topics = set()
_level_order = {"FOUNDATION": 0, "BEGINNER": 1, "INTERMEDIATE": 2, "ADVANCED": 3, "CAPSTONE": 4}
_learning_order = {
    "autodl": "foundations tensors neurons layers activation forward loss backprop optimizers learning-rate training regularization cnn transfer pytorch image-capstone".split(),
    "autonlp": "foundations cleaning preprocessing stopwords word-forms bow tfidf ngrams pipeline metrics embeddings context attention transformers baseline text-capstone".split(),
    "python": "basics operators conditions loops collections maps-sets parameters files-errors numpy pandas filter-group plotting reusable models".split(),
    "sql": "relations select sort-distinct aggregates groups left-join joins subqueries ctes case-null windows revenue business".split(),
    "eda": "purpose schema types quality categories outliers charts relationships transformations insights".split(),
    "genai": "llm tokens temperature hallucinations embeddings retrieval rag citations evaluation agents guardrails".split(),
    "prompting": "clarity context format variables few-shot extraction grounding reduce-hallucination compare evaluation tool-agent".split(),
}
def _place_topic(item):
    if item["id"] in _visited_topics:
        return
    if item["id"] in _visiting_topics:
        raise ValueError("Circular Learning prerequisites are not allowed.")
    _visiting_topics.add(item["id"])
    for parent in item["prerequisites"]:
        if parent not in _topic_index:
            raise ValueError(f"Unknown Learning prerequisite: {parent}")
        _place_topic(_topic_index[parent])
    _visiting_topics.remove(item["id"])
    _visited_topics.add(item["id"])
    _ordered_topics.append(item)

for _module in LABS:
    _group = [item for item in TOPICS if item["module"] == _module]
    if _module != "automl":
        _positions = {f"{_module}-{key}": index for index, key in enumerate(_learning_order[_module])}
        _group.sort(key=lambda item: (_positions.get(item["id"], 100), _level_order[item["level"]]))
    for _unit in _group:
        _place_topic(_unit)
TOPICS = _ordered_topics

BY_ID = {item["id"]: item for item in TOPICS}

# Optional educational illustrations. These are client-side explanations, not lab settings.
LESSON_VISUALS = {
    "automl-features-target": ("Clues and answer", "Features are clues; the target is the answer the model learns to predict.", "PROCESS_FLOW"),
    "automl-splits": ("Study and exam", "Training rows are study material; held-out rows are an unseen exam.", "TRAIN_TEST_SPLIT"),
    "automl-foundations-depth": ("Memorising an exam", "Overfitting is like memorising past exam questions instead of understanding the subject.", "TRAIN_TEST_SPLIT"),
    "automl-imbalance": ("Screening decisions", "Medical screening values finding cases; security screening also weighs false alarms.", "PRECISION_RECALL"),
    "automl-metrics": ("Different costs of mistakes", "Precision asks how often an alert is right; recall asks how many true cases were found.", "CONFUSION_MATRIX"),
    "automl-clustering": ("Grouping shoppers", "Customers with similar buying behaviour can form useful groups without predefined labels.", "CLUSTERING"),
    "automl-clustering-depth": ("Grouping shoppers", "A segment is useful only when its members share meaningful traits.", "CLUSTERING"),
    "autodl-neurons": ("A team of decision makers", "Each neuron weighs incoming clues; layers pass increasingly refined decisions onward.", "NEURAL_NETWORK"),
    "autodl-training": ("Walking downhill", "The learning rate sets step size downhill; an epoch is one full pass through the training set.", "GRADIENT_DESCENT"),
    "autodl-cnn": ("Edges to objects", "A CNN first notices edges, then shapes and parts that help recognize an object.", "CNN_KERNEL"),
    "autonlp-preprocessing": ("Sentence building blocks", "Tokenization separates a sentence into units the model can process.", "TOKENIZATION"),
    "autonlp-tfidf": ("Distinctive words", "A word matters more when frequent in one document but uncommon across all documents.", "PROCESS_FLOW"),
    "autonlp-transformers": ("Meaning maps and focused readers", "Embeddings place similar meanings nearby; attention highlights useful word relationships as readers examine them in parallel.", "ATTENTION"),
}
for lesson_id, (analogy_title, analogy_description, visual_type) in LESSON_VISUALS.items():
    if lesson_id in BY_ID:
        BY_ID[lesson_id]["analogy"] = {"title": analogy_title, "description": analogy_description}
        BY_ID[lesson_id]["visual"] = {"type": visual_type, "config": {}}

# Short teaching examples belong to a lesson, not to a whole lab module.
# Variables such as df, X_train and model denote learner-prepared data or models.
LESSON_EXAMPLES = {
    "automl-features-target": ("python", "Separate inputs and outcome", 'X = df.drop(columns=["id", "stroke"])\ny = df["stroke"]', "Exclude an identifier when it cannot generalize."),
    "automl-splits": ("python", "Keep an unseen test set", 'from sklearn.model_selection import train_test_split\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y\n)', "Use stratify for classification when each class has enough rows; omit it for regression."),
    "automl-imbalance": ("python", "Inspect missed positive cases", 'from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score\nprint(confusion_matrix(y_test, y_pred))\nprint(recall_score(y_test, y_pred, pos_label=1))\npositive_scores = model.predict_proba(X_test)[:, 1]\nprint(roc_auc_score(y_test, positive_scores))', "The second probability column belongs to model.classes_[1]; verify that this is the intended positive class."),
    "automl-regression-errors": ("python", "Compare error measures", 'from sklearn.metrics import mean_absolute_error, mean_squared_error\nmae = mean_absolute_error(y_test, y_pred)\nrmse = mean_squared_error(y_test, y_pred) ** 0.5', "RMSE weighs large misses more heavily than MAE."),
    "automl-clustering": ("python", "Group unlabeled rows", 'from sklearn.cluster import KMeans\nclusters = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)', "Clustering uses prepared features, without a target y."),
    "automl-preparation": ("python", "Avoid target leakage", 'leaky = df.drop(columns=["churn"])  # Still includes post-outcome fields\nX = df.drop(columns=["churn", "account_closed", "churn_reason", "customer_id"])', "The first line is intentionally unsafe; remove leaked fields before splitting."),
    "automl-metrics": ("python", "Measure classification outcomes", 'from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\nprint(accuracy_score(y_test, y_pred))\nprint(precision_score(y_test, y_pred, zero_division=0))\nprint(recall_score(y_test, y_pred, zero_division=0))\nprint(f1_score(y_test, y_pred, zero_division=0))', "For multiclass tasks, choose an appropriate averaging method and class labels."),
    "automl-explainability": ("python", "Read fitted feature influence", 'for name, value in zip(feature_names, model.feature_importances_):\n    print(name, value)', "Only estimators exposing feature_importances_ support this illustration; influence is not causation."),
    "automl-foundations-depth": ("pseudocode", "Recognize overfitting", 'Training score: 0.99\nUnseen test score: 0.62\nLarge gap -> check whether the model memorized training details', "Like studying only past exam answers, a high training score need not transfer to unseen cases."),
    "automl-preparation-depth": ("python", "Fit preparation on training rows", 'from sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nimputer = SimpleImputer(strategy="median")\nX_train_numeric = imputer.fit_transform(X_train_numeric)\nX_test_numeric = imputer.transform(X_test_numeric)\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train_numeric)\nX_test_scaled = scaler.transform(X_test_numeric)\nencoder = OneHotEncoder(handle_unknown="ignore")', "Fit categorical encoding on training categories too; use a column pipeline for mixed types."),
    "automl-classification-depth": ("python", "Fit a simple classifier", 'from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(max_iter=1000)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)', "Compare this baseline with other classifiers on the same held-out rows."),
    "automl-regression-depth": ("python", "Predict a numeric target", 'from sklearn.linear_model import Ridge\nmodel = Ridge(alpha=1.0)\nmodel.fit(X_train, y_train)\nprice_pred = model.predict(X_test)', "Use a regression metric such as MAE on held-out prices."),
    "automl-clustering-depth": ("python", "Compare candidate cluster counts", 'from sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\nfor k in (2, 3, 4):\n    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)\n    print(k, silhouette_score(X_scaled, labels))', "Silhouette requires enough rows and at least two nonempty clusters."),
    "automl-advanced-depth": ("python", "Keep tuning separate from final test", 'from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import cross_val_score\nmodel = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)\nprint(cross_val_score(model, X_train, y_train, cv=5).mean())', "Cross-validation uses training rows; reserve the final test split for evaluation."),
    "autodl-neurons": ("python", "One weighted transformation", 'import torch\nx = torch.tensor([1.0, 2.0])\nweights = torch.tensor([0.5, -0.25])\nbias = torch.tensor(0.1)\noutput = x @ weights + bias\nprint(output)', "The visual shows several connected neurons; this computes one neuron before activation."),
    "autodl-training": ("python", "Activation, loss, and update", 'import torch\nlogits = torch.tensor([[1.0, 2.0]], requires_grad=True)\nlabels = torch.tensor([1])\nprint(torch.relu(logits))  # Activation illustration\ncriterion = torch.nn.CrossEntropyLoss()\noptimizer = torch.optim.SGD([logits], lr=0.1)\noptimizer.zero_grad()\nloss = criterion(logits, labels)\nloss.backward()\noptimizer.step()', "This optimizes a tiny illustrative tensor; a real network updates model parameters over batches and epochs."),
    "autodl-cnn": ("python", "A sliding image filter", 'import torch\nconvolution = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)\npooling = torch.nn.MaxPool2d(2)', "Conv2d learns local patterns; pooling reduces the feature-map size."),
    "autodl-pytorch": ("python", "One batch update", 'for inputs, labels in train_loader:\n    optimizer.zero_grad()\n    logits = model(inputs)\n    loss = criterion(logits, labels)\n    loss.backward()\n    optimizer.step()', "Repeat over batches and epochs; evaluate separately without updating weights."),
    "autonlp-preprocessing": ("python", "Tokens and negation", 'text = "not good"\ntokens = text.lower().split()\nstopwords = {"the", "a", "is"}\nkept = [token for token in tokens if token not in stopwords]\nprint(kept)  # Keep "not" for sentiment', "Whitespace splitting is illustrative. Stemming may reduce 'running' to 'run'; lemmatization may map 'was' to 'be'."),
    "autonlp-tfidf": ("python", "Count vectors versus TF-IDF", 'from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer\ntexts = ["urgent support ticket", "routine support ticket"]\ncounts = CountVectorizer().fit_transform(texts)\ntfidf = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(texts)\nprint(counts.shape, tfidf.shape)', "An n-gram range of (1, 2) includes single words and two-word phrases."),
    "autonlp-metrics": ("python", "Inspect each text label", 'from sklearn.metrics import classification_report, confusion_matrix\nprint(classification_report(y_test, y_pred, zero_division=0))\nprint(confusion_matrix(y_test, y_pred))', "Per-class results reveal categories hidden by overall accuracy."),
    "autonlp-transformers": ("pseudocode", "Compare before adding complexity", 'Tokens -> meaning vectors (embeddings) -> attention across tokens\nCompare TF-IDF baseline and transformer on the same held-out texts\nChoose only if the improvement justifies data and compute cost', "Conceptual flow; attention here is illustrative, not a measured model weight or API call."),
    "python-basics": ("python", "Values, branch, and loop", 'customer_count = 3\nfor index in range(customer_count):\n    if index == 0:\n        print("First customer")\n    else:\n        print(index)', "Readable names make control flow easier to follow."),
    "python-collections": ("python", "Use collections in a function", 'customers = [{"name": "Asha", "spend": 120}, {"name": "Ravi", "spend": 40}]\ndef high_spenders(rows, minimum=100):\n    return [row["name"] for row in rows if row["spend"] >= minimum]\nprint(high_spenders(customers))', "The list contains dictionaries; the function returns a filtered list."),
    "python-pandas": ("python", "Clean and group sales", 'import pandas as pd\nsales = pd.DataFrame({"region": ["North", "North"], "amount": [10, None]})\nsales["amount"] = sales["amount"].fillna(0)\nprint(sales.groupby("region")["amount"].sum())', "Inspect missing amounts before deciding whether zero is an appropriate replacement."),
    "python-files-errors": ("python", "Read a file safely", 'from pathlib import Path\npath = Path("example.txt")\ntry:\n    with path.open(encoding="utf-8") as file:\n        text = file.read()\nexcept FileNotFoundError:\n    print("Choose an existing file")', "The context manager closes the file; catch the expected error specifically."),
    "python-models": ("python", "Keep preprocessing with a model", 'from sklearn.pipeline import make_pipeline\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.linear_model import LogisticRegression\nmodel = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=1000))\nmodel.fit(X_train, y_train)\nprint(model.score(X_test, y_test))', "This numeric-feature example fits imputation only on training data."),
    "sql-select": ("sql", "Select and filter customers", "SELECT name, city\nFROM customers\nWHERE city = 'Chennai'\nORDER BY name;", "Uses the customers(name, city) learning challenge fixture."),
    "sql-groups": ("sql", "Count employees by department", "SELECT department, COUNT(*) AS employee_count\nFROM employees\nGROUP BY department\nHAVING COUNT(*) >= 1;", "Uses the employees(department) learning challenge fixture."),
    "sql-joins": ("sql", "Join the customer revenue fixture", "WITH customer_revenue AS (\n  SELECT c.name, p.price * o.quantity AS amount\n  FROM customers c\n  JOIN orders o ON o.customer_id = c.id\n  JOIN products p ON p.id = o.product_id\n)\nSELECT name, SUM(amount) AS total_revenue\nFROM customer_revenue GROUP BY name;", "Uses the customers, orders, and products learning challenge fixture."),
    "sql-windows": ("sql", "Rank each customer's orders", "SELECT customer_id, quantity,\n       ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY quantity DESC) AS order_rank\nFROM orders;", "Uses the orders(customer_id, product_id, quantity) learning challenge fixture."),
    "sql-revenue": ("sql", "Calculate customer revenue", "SELECT c.name, SUM(p.price * o.quantity) AS total_revenue\nFROM customers c\nJOIN orders o ON o.customer_id = c.id\nJOIN products p ON p.id = o.product_id\nGROUP BY c.name;", "Uses the fixed customer-revenue learning challenge fixture."),
    "eda-schema": ("python", "Inspect shape and types", 'print(df.shape)\nprint(df.dtypes)\nprint(df.head())', "df represents the dataset selected for analysis."),
    "eda-quality": ("python", "Count missing and duplicate rows", 'print(df.isna().sum())\nprint(df.duplicated().sum())', "Measure quality before changing or dropping rows."),
    "eda-charts": ("python", "Compare distributions", 'import matplotlib.pyplot as plt\ndf.boxplot(column="salary", by="department")\nplt.suptitle("")\nplt.show()', "Use a boxplot for numeric salary across department categories."),
    "eda-relationships": ("python", "Plot before interpreting correlation", 'import matplotlib.pyplot as plt\nprint(df[["price", "area"]].corr())\ndf.plot.scatter(x="area", y="price")\nplt.show()', "Correlation does not establish causation; inspect outliers in the scatter."),
    "eda-insights": ("prompt", "State an evidence-based insight", 'Observation: Sales increased in two regions.\nLimit: Some stores have missing records.\nNext check: Compare matched stores across the same dates.', "Use actual observed values when writing your own insight."),
    "genai-llm": ("pseudocode", "Tokens and context", 'Input text -> tokens -> context window -> next-token prediction\nFluent output -> verify claims against evidence', "Conceptual flow, not a model call or tokenizer implementation."),
    "genai-rag": ("pseudocode", "Ground an answer", 'User question -> retrieve selected document passages\nPassages + question -> generate answer\nAnswer -> cite supported passages; state when evidence is insufficient', "Conceptual RAG flow; no invented NxZenAI API calls."),
    "genai-agents": ("json", "Describe a complete tool request", '{"tool": "lookup_order", "arguments": {"order_id": "A123"}}', "Illustrative tool contract only; invoke a real tool only when the user's task requires it."),
    "prompting-clarity": ("prompt", "Give a clear task", 'You are a tutor. Explain precision to a first-year student in three bullets, with one example.', "Specify task, audience, format, and a useful constraint."),
    "prompting-grounding": ("prompt", "Restrict the evidence", 'Answer using only the selected project documents. Cite each supported claim. If the documents do not answer, say what is missing.', "Select source documents before running this prompt."),
    "prompting-compare": ("prompt", "Compare under the same criteria", 'Task: Summarize the same document in five bullets.\nCompare outputs for factual support, completeness, format, and latency.', "Keep the task and criteria fixed across saved runs; do not assume a longer answer wins."),
    "automl-classification-capstone": ("pseudocode", "Classification case sequence", 'Define positive class and decision cost -> train on the case dataset\nReview recall and false negatives -> make a recommendation', "Illustrative capstone sequence, not a reconstruction of the saved run."),
    "automl-regression-capstone": ("pseudocode", "Regression case sequence", 'Define target price and acceptable error -> train\nReview MAE, RMSE and residuals -> make a recommendation', "Illustrative capstone sequence, not a reconstruction of the saved run."),
    "automl-clustering-capstone": ("pseudocode", "Segmentation case sequence", 'Choose customer features, without a target -> compare groups\nInspect profiles and sizes -> recommend a business action', "Illustrative capstone sequence, not a reconstruction of the saved run."),
    "autodl-image-capstone": ("pseudocode", "Image case sequence", 'Inspect labeled image classes -> train and validate\nReview class errors -> try inference -> recommend an action', "Illustrative capstone sequence, not a reconstruction of the saved run."),
    "autonlp-text-capstone": ("pseudocode", "Text case sequence", 'Inspect ticket text and labels -> prepare text and train\nReview per-label errors -> try inference -> recommend an action', "Illustrative capstone sequence, not a reconstruction of the saved run."),
}
for lesson_id, (kind, title, content, note) in LESSON_EXAMPLES.items():
    BY_ID[lesson_id]["illustrative_example"] = {"type": kind, "title": title, "content": content, "note": note}

# A topic requires only evidence that its learner workflow can actually produce.
# Other categories remain visible as optional learning feedback.
for item in TOPICS:
    required = {"theory": 70}
    if item.get("is_capstone"):
        item["mastery_requirements"] = {"practical": 70}
        continue
    if item["module"] in {"automl", "autodl", "autonlp"}:
        if item.get("reasoning_check"):
            required.update(reasoning=70, quiz=70)
        if item.get("code_check"):
            required["code_reading"] = 70
        item["mastery_requirements"] = required
        continue
    if item.get("reasoning_check"):
        required.update(reasoning=70, quiz=70)
    if item.get("code_check"):
        required["code_reading"] = 70
    if item["id"] in {"python-basics", "python-collections", "python-pandas"}:
        required["coding"] = 70
    if item["module"] == "sql" and item["id"] in {"sql-select", "sql-groups", "sql-joins"}:
        required["coding"] = 70
    if item["id"] in {"eda-quality", "eda-charts", "eda-relationships", "eda-insights",
                      "genai-rag", "genai-agents", "prompting-grounding", "prompting-compare"}:
        required["practical"] = 70
    if item["id"] in {"genai-rag", "prompting-compare"}:
        required["case"] = 70
    item["mastery_requirements"] = required

# Grading specifications stay server-side. Never serialize hidden tests or expected rows.
PYTHON_CHALLENGES = {
    "automl-features-target": {"instruction": "Implement split_xy(df, target): return input columns X and target Series y without changing df.", "starter_code": "def split_xy(df, target):\n    # Return X, y.\n    pass\n", "hidden_tests": ["from pandas import DataFrame; d=DataFrame({'age':[20,30],'target':[0,1]}); X,y=split_xy(d,'target'); assert list(X.columns)==['age'] and y.tolist()==[0,1]", "from pandas import DataFrame; d=DataFrame({'id':[1],'x':[2],'label':[3]}); X,y=split_xy(d,'label'); assert list(X.columns)==['id','x'] and y.tolist()==[3]"], "score": 100, "max_attempts": 10},
    "automl-splits": {"instruction": "Implement holdout_rows(X, y) using train_test_split with test_size=0.25 and random_state=42; return X_train, X_test, y_train, y_test.", "starter_code": "from sklearn.model_selection import train_test_split\n\ndef holdout_rows(X, y):\n    # Keep held-out rows separate.\n    pass\n", "hidden_tests": ["a,b,c,d=holdout_rows(list(range(8)),list(range(8))); assert len(a)==6 and len(b)==2 and set(a).isdisjoint(b) and a==c and b==d", "a,b,c,d=holdout_rows(list(range(12)),list(range(12))); assert len(b)==3 and set(list(a)+list(b))==set(range(12))"], "score": 100, "max_attempts": 10},
    "automl-metrics": {"instruction": "Implement positive_recall(y_true, y_pred) using sklearn.metrics.recall_score.", "starter_code": "from sklearn.metrics import recall_score\n\ndef positive_recall(y_true, y_pred):\n    # Return positive-class recall.\n    pass\n", "hidden_tests": ["assert positive_recall([1,1,0,0],[1,0,0,0]) == 0.5", "assert positive_recall([1,1,0],[1,1,1]) == 1.0"], "score": 100, "max_attempts": 10},
    "python-basics": {"instruction": "Return twice the given number, including negative values and zero.", "starter_code": "def double(value):\n    # Return twice the input.\n    pass\n", "hidden_tests": ["assert double(2) == 4", "assert double(-3) == -6", "assert double(0) == 0"], "score": 100, "max_attempts": 10},
    "python-collections": {"instruction": "Return how many distinct values are in a list.", "starter_code": "def unique_count(values):\n    # Count distinct values.\n    pass\n", "hidden_tests": ["assert unique_count([1, 1, 2]) == 2", "assert unique_count([]) == 0", "assert unique_count(['a', 'b', 'a']) == 2"], "score": 100, "max_attempts": 10},
    "python-pandas": {"instruction": "Sum amount by region for a list of sale dictionaries.", "starter_code": "def revenue_by_region(rows):\n    # Return a dictionary of region totals.\n    pass\n", "hidden_tests": ["assert revenue_by_region([{'region': 'A', 'amount': 2}, {'region': 'A', 'amount': 3}]) == {'A': 5}", "assert revenue_by_region([]) == {}"], "score": 100, "max_attempts": 10},
}
SQL_CHALLENGES = {
    "sql-select": {"instruction": "Return customer names and cities in columns named name and city.", "starter_query": "SELECT name FROM customers", "fixture_schema": "customers(name TEXT, city TEXT); rows: Rahul/Hyderabad, Sneha/Bangalore, Amit/Chennai. Grading uses these fixed rows even if your SQL Lab database changes.", "fixture_cte": "WITH customers(name, city) AS (VALUES ('Rahul','Hyderabad'),('Sneha','Bangalore'),('Amit','Chennai')) ", "expected_columns": ["name", "city"], "expected_rows": [["Amit", "Chennai"], ["Rahul", "Hyderabad"], ["Sneha", "Bangalore"]], "order_sensitive": False, "score": 100, "max_attempts": 10},
    "sql-groups": {"instruction": "Count employees per department. Name the count column employee_count.", "starter_query": "SELECT department FROM employees", "fixture_schema": "employees(department TEXT); four rows: AI, Data Science, Engineering, Sales. Grading uses these fixed rows even if your SQL Lab database changes.", "fixture_cte": "WITH employees(department) AS (VALUES ('AI'),('Data Science'),('Engineering'),('Sales')) ", "expected_columns": ["department", "employee_count"], "expected_rows": [["AI", 1], ["Data Science", 1], ["Engineering", 1], ["Sales", 1]], "order_sensitive": False, "score": 100, "max_attempts": 10},
    "sql-joins": {"instruction": "Join customers, orders, and products. Return customer name and total_revenue (price times quantity), grouped by customer.", "starter_query": "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id", "fixture_schema": "customers(id,name): (1,Rahul),(2,Sneha). products(id,price): (1,10),(2,20). orders(customer_id,product_id,quantity): (1,1,2),(1,2,1),(2,2,3). Grading uses these fixed rows.", "fixture_cte": "WITH customers(id,name) AS (VALUES (1,'Rahul'),(2,'Sneha')), products(id,price) AS (VALUES (1,10),(2,20)), orders(customer_id,product_id,quantity) AS (VALUES (1,1,2),(1,2,1),(2,2,3)) ", "expected_columns": ["name", "total_revenue"], "expected_rows": [["Rahul", 40], ["Sneha", 60]], "order_sensitive": False, "score": 100, "max_attempts": 10},
}

CASES = [
    {"id": key, "module": module, "title": title, "problem": problem,
     "steps": steps, "lab_url": LABS[module][1], "competencies": competencies}
    for key, module, title, problem, steps, competencies in [
        ("stroke-risk", "automl", "Stroke Risk", "An imbalanced stroke dataset has high accuracy but low recall. Decide how to evaluate and improve it.", ["Inspect data", "Choose metric", "Train in AutoML", "Interpret PR and confusion matrix", "Present decision"], ["recall", "imbalance"]),
        ("fraud-detection", "automl", "Fraud Detection", "Detect rare fraud while explaining false-positive costs.", ["Inspect classes", "Train", "Compare PR", "Interpret errors"], ["precision", "recall"]),
        ("customer-churn", "automl", "Customer Churn Decision System", "Connect data quality, modeling, predictions, and business recommendations.", ["EDA", "Python or SQL", "AutoML", "Interpret", "Predict", "Recommend"], ["data quality", "classification"]),
        ("image-classification", "autodl", "Image Classification", "Check image data reliability before interpreting a classifier.", ["Inspect classes", "Train", "Check readiness", "Predict"], ["validation", "generalization"]),
        ("defect-detection", "autodl", "Defect Detection", "Evaluate errors for visual product defects.", ["Inspect images", "Train", "Review per-class errors"], ["false negatives", "vision"]),
        ("financial-sentiment", "autonlp", "Financial Sentiment", "Classify local financial headlines and discuss ambiguous language.", ["Inspect labels", "Train", "Evaluate", "Explain limitations"], ["sentiment", "evaluation"]),
        ("support-tickets", "autonlp", "Support Ticket Classification", "Route support requests while checking minority intents.", ["Inspect text", "Train", "Review per-class metrics"], ["text", "imbalance"]),
        ("retail-sales", "eda", "Retail Sales Analysis", "Find trends and data-quality issues in sales records.", ["Inspect", "Clean", "Visualize", "Report"], ["missingness", "charts"]),
        ("customer-revenue", "sql", "Customer Revenue Analysis", "Build a grouped revenue query and explain join cardinality.", ["Inspect tables", "Join", "Aggregate", "Explain"], ["joins", "grouping"]),
        ("enterprise-rag", "genai", "Enterprise RAG Assistant", "Use project documents to answer with verified citations.", ["Select documents", "Design prompt", "Ask", "Verify sources"], ["retrieval", "grounding"]),
        ("support-prompt", "prompting", "Customer Support Prompt Design", "Design and compare grounded support responses.", ["Define rubric", "Write prompt", "Run two models", "Compare"], ["constraints", "evaluation"]),
    ]
]
CASE_BY_ID = {case["id"]: case for case in CASES}

# Concrete inputs and the result that must be verified; checkboxes alone are progress.
CASE_INPUTS = {
    "stroke-risk": ("automl-classification-capstone", "AUTOML_RUN", "Upload a labeled tabular dataset with target column stroke (0=no stroke, 1=stroke); select Classification and inspect recall, PR, and confusion matrix."),
    "fraud-detection": ("automl-classification-capstone", "AUTOML_RUN", "Upload tabular transactions with numeric/categorical features and a fraud target (0=legitimate, 1=fraud); use Classification and inspect positive-class errors."),
    "customer-churn": ("automl-classification-capstone", "AUTOML_RUN", "Use a customer table with a binary churn target. Inspect it in EDA, prepare features in Python or SQL if needed, then train Classification in AutoML and save the result."),
    "image-classification": ("autodl-image-capstone", "AUTODL_RUN", "Upload a ZIP of labeled images arranged in at least two class folders; train Image Classification and review validation results."),
    "defect-detection": ("autodl-image-capstone", "AUTODL_RUN", "Upload a ZIP with normal and defect class folders. Train Image Classification and inspect errors; this case does not require object detection."),
    "financial-sentiment": ("autonlp-text-capstone", "AUTONLP_RUN", "Upload CSV text with a headline column and a sentiment label column; select Sentiment analysis and train a saved model."),
    "support-tickets": ("autonlp-text-capstone", "AUTONLP_RUN", "Upload CSV tickets with a text column and a category/intent target; select Text classification or Intent classification."),
    "retail-sales": ("eda-schema", "EDA_REPORT", "Upload sales rows with a date, product/category, quantity, and amount. Inspect quality and trends, then generate an EDA report."),
    "customer-revenue": ("sql-joins", "SQL_EXECUTION", "Use the fixed SQL customer-revenue challenge: join customers(id,name), orders(customer_id,product_id,quantity), and products(id,price); return name and total_revenue, then submit the verified query."),
    "enterprise-rag": ("genai-rag", "GENAI_ACTIVITY", "Create a GenAI project, add and select your own reference documents, ask a document-grounded question, then verify a persisted answer with citations."),
    "support-prompt": ("prompting-compare", "PROMPT_LAB_RUN", "Create a support prompt in Prompt Lab, run it successfully, and compare saved runs if multiple models are available."),
}
for case in CASES:
    case["topic_id"], case["evidence_type"], case["input_requirements"] = CASE_INPUTS[case["id"]]
    if case["id"] in {"stroke-risk", "fraud-detection", "customer-churn"}:
        case["expected_task"] = "classification"
        case["expected_target"] = {"stroke-risk": "stroke", "fraud-detection": "fraud", "customer-churn": "churn"}[case["id"]]
    if case["id"] in {"image-classification", "defect-detection"}:
        case["compatible_tasks"] = ["image_classification"]
    if case["id"] == "financial-sentiment":
        case["compatible_tasks"] = ["sentiment_analysis"]
    if case["id"] == "support-tickets":
        case["compatible_tasks"] = ["text_classification", "intent_classification"]

# Bounded assessment bank. Choices/keys stay on the server until submission.
ACTIVITIES = [
    {"id": "ml-stroke-case", "module": "automl", "topic_id": "automl-imbalance", "type": "case_study", "category": "case", "prompt": "A stroke model is 97% accurate but detects only 30% of stroke cases. What is the soundest next step?", "choices": ["Deploy because accuracy is high", "Inspect false negatives and optimize recall with PR diagnostics", "Drop the positive class"], "correct_indices": [1], "explanation": "High accuracy can mask missed positive cases."},
    {"id": "ml-metrics-multi", "module": "automl", "topic_id": "automl-imbalance", "type": "multi_select", "category": "theory", "prompt": "Which diagnostics help assess a rare positive class? Select all that apply.", "choices": ["Confusion matrix", "Precision–Recall curve", "Training row count alone", "Positive-class recall"], "correct_indices": [0, 1, 3], "explanation": "Use errors and positive-class tradeoffs, not dataset size alone."},
    {"id": "python-output", "module": "python", "topic_id": "python-collections", "type": "output_prediction", "category": "coding", "prompt": "What is printed by print(len(set([1, 1, 2])))?", "choices": ["1", "2", "3"], "correct_indices": [1], "explanation": "A set keeps unique values: 1 and 2."},
    {"id": "python-debug", "module": "python", "topic_id": "python-pandas", "type": "debugging", "category": "coding", "prompt": "Which expression returns revenue totals by region?", "choices": ["sales.groupby('region')['amount'].sum()", "sales.groupby('region')['amount']", "sales.head()"], "correct_indices": [0], "explanation": "Grouping needs an aggregation such as sum()."},
    {"id": "python-complete", "module": "python", "topic_id": "python-collections", "type": "complete_missing_code", "category": "coding", "prompt": "Complete: return [c for c in customers if c['spend'] ___ limit] to keep high-spend customers.", "choices": [">", "<", "=="], "correct_indices": [0], "explanation": "The threshold condition should select spend above the limit."},
    {"id": "python-duplicates", "module": "python", "topic_id": "python-pandas", "type": "business_coding", "category": "coding", "prompt": "Which pandas expression identifies duplicate customer IDs?", "choices": ["df.duplicated(subset=['customer_id'])", "df.head()", "df.dropna()"], "correct_indices": [0], "explanation": "duplicated checks repeated keys; it does not remove them."},
    {"id": "sql-join", "module": "sql", "topic_id": "sql-joins", "type": "query_correction", "category": "coding", "prompt": "Which join condition safely joins orders to customers?", "choices": ["ON orders.customer_id = customers.id", "ON orders.id = customers.id", "No ON clause"], "correct_indices": [0], "explanation": "Join on the foreign key and matching customer key."},
    {"id": "sql-revenue", "module": "sql", "topic_id": "sql-revenue", "type": "business_problem", "category": "coding", "prompt": "How does the fixed customer-revenue challenge calculate total_revenue?", "choices": ["Join orders and products, then SUM(price * quantity) by customer", "GROUP BY region with SUM(amount)", "ORDER BY city only"], "correct_indices": [0], "explanation": "The fixture has customers, orders and products; it has no region field."},
    {"id": "eda-chart", "module": "eda", "topic_id": "eda-charts", "type": "result_interpretation", "category": "case", "prompt": "You need to compare salary distributions across departments. Which chart is most informative?", "choices": ["Boxplot", "ROC curve", "Pie chart"], "correct_indices": [0], "explanation": "A boxplot compares center and spread across categories."},
    {"id": "dl-test", "module": "autodl", "topic_id": "autodl-cnn", "type": "case_study", "category": "case", "prompt": "Training accuracy is high but the independent image test score is poor. What should you inspect first?", "choices": ["Dataset shift, leakage, and per-class coverage", "Only training loss", "Increase reported confidence"], "correct_indices": [0], "explanation": "Independent test data reveals generalization risks."},
    {"id": "nlp-baseline", "module": "autonlp", "topic_id": "autonlp-transformers", "type": "case_study", "category": "case", "prompt": "Should a small support-ticket task automatically use a transformer?", "choices": ["Yes, always", "Compare a simple baseline on held-out data first", "Skip evaluation"], "correct_indices": [1], "explanation": "Model complexity should be justified by evidence."},
    {"id": "rag-grounding", "module": "genai", "topic_id": "genai-rag", "type": "result_interpretation", "category": "case", "prompt": "Retrieved documents do not support an answer. What should a grounded assistant do?", "choices": ["State that evidence is insufficient", "Invent a citation", "Use unrelated memory as fact"], "correct_indices": [0], "explanation": "Grounded answers must acknowledge missing evidence."},
    {"id": "prompt-debugger", "module": "prompting", "topic_id": "prompting-clarity", "type": "debugging", "category": "theory", "prompt": "What is missing from the prompt 'Analyze this'?", "choices": ["Only a model name", "Objective, context, constraints, and output format", "More punctuation"], "correct_indices": [1], "explanation": "A useful prompt makes its task and success criteria explicit."},
    {"id": "prompt-ab", "module": "prompting", "topic_id": "prompting-compare", "type": "case_study", "category": "case", "prompt": "How should two prompts be compared fairly?", "choices": ["Same task and explicit criteria", "Different tasks", "Pick the longest response"], "correct_indices": [0], "explanation": "Compare against consistent criteria and evidence."},
]
ACTIVITY_BY_ID = {item["id"]: item for item in ACTIVITIES}

# Add teaching metadata without changing topic IDs, prerequisites, graded items or evidence rules.
from .learner_lifecycle import learner_lifecycle
for _item in TOPICS:
    if _item["module"] == "automl":
        _task = _item.get("expected_task") or ("clustering" if "clustering" in _item.get("stage", "").lower() else "regression" if "regression" in _item.get("stage", "").lower() else "classification")
        _item["learner_lifecycle"] = learner_lifecycle(_task)
    if _item["id"] == "automl-foundations-depth":
        _item["theory_sections"].append({"heading": "The wider Data Science lifecycle", "text": "Start with the business decision and data collection/provenance, then cleaning, EDA, feature engineering/selection and a split strategy for supervised tasks. Select candidates, fit parameters, tune with validation, evaluate, analyze errors, run inference and recommend next steps. The process is iterative. Clustering uses an unsupervised strategy without a labelled target or mandatory supervised split; inspect scaling and cluster profiles instead."})

# Explicit curricular placement: presentation metadata only, never prerequisite order.
ML_TOPIC_PLACEMENT = {
    "automl-foundations-depth": ("problem", ["business"], None),
    "automl-features-target": ("data", ["ingestion", "features"], None),
    "automl-preparation": ("preparation", ["cleaning", "features"], None),
    "automl-preparation-depth": ("preparation", ["cleaning", "features"], None),
    "automl-splits": ("modeling", ["split"], None),
    "automl-classification-depth": ("training", ["selection", "training", "evaluation"], "classification"),
    "automl-regression-depth": ("training", ["selection", "training", "errors"], "regression"),
    "automl-clustering-depth": ("training", ["strategy", "selection", "parameters", "profiling"], "clustering"),
    "automl-imbalance": ("evaluation", ["evaluation", "errors"], "classification"),
    "automl-regression-errors": ("evaluation", ["evaluation", "errors"], "regression"),
    "automl-clustering": ("evaluation", ["parameters", "evaluation", "profiling"], "clustering"),
    "automl-metrics": ("evaluation", ["evaluation"], None),
    "automl-explainability": ("evaluation", ["errors"], None),
    "automl-advanced-depth": ("evaluation", ["tuning", "evaluation", "errors"], None),
}
for _item in TOPICS:
    if _item["module"] != "automl":
        continue
    _placement = (("apply", ["recommendation"], _item.get("expected_task")) if _item.get("is_capstone")
                  else ML_TOPIC_PLACEMENT.get(_item["id"], ("additional", [], None)))
    _item["roadmap_group"], _item["lifecycle_stages"], _item["learning_task"] = _placement

# Shared presentation metadata leaves content, IDs and prerequisite rules intact.
from .mastery_roadmaps import attach_topic_metadata
for _item in TOPICS:
    attach_topic_metadata(_item, LABS[_item["module"]][1])
