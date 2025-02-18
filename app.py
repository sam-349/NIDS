from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Metrics data for models
metrics_data = {
    "Accuracy": {"SVM": 85, "DT": 80, "KNN": 78},
    "Precision": {"SVM": 86, "DT": 82, "KNN": 79},
    "Recall": {"SVM": 84, "DT": 79, "KNN": 76},
    "F1 Score": {"SVM": 85, "DT": 80, "KNN": 77}
}

# Dataset file paths (dummy CSV files needed)
datasets = {
    "KDDCUP99": "static/datasets/KDDCUP99.csv",
    "NSL-KDD": "static/datasets/NSL-KDD.csv"
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/models', methods=['GET', 'POST'])
def models():
    selected_metric = request.form.get('metric', 'Accuracy')
    selected_dataset = request.form.get('dataset', 'KDDCUP99')

    # Generate graph for selected metric
    metric_values = metrics_data[selected_metric]

    plt.figure(figsize=(6, 4))
    plt.bar(metric_values.keys(), metric_values.values(), color=['blue', 'green', 'red'])
    plt.xlabel("ML Models")
    plt.ylabel(selected_metric)
    plt.title(f"{selected_metric} for ML Models")
    graph_path = os.path.join("static", "graph.png")
    plt.savefig(graph_path)
    plt.close()

    # Load dataset
    dataset_path = datasets[selected_dataset]
    df = pd.read_csv(dataset_path).head(10)  # Show first 10 rows

    return render_template('models.html', metric=selected_metric, dataset=selected_dataset, graph=graph_path, table=df.to_html(classes='table table-striped'))

if __name__ == '__main__':
    app.run(debug=True)
