import pandas as pd
import matplotlib.pyplot as plt

# Supposons que X_test et y_test sont déjà définis
# Supposons que best_models est un dictionnaire contenant vos modèles optimisés

def evaluate_on_test_set(best_models, X_test, y_test):
    results = []

    for model_name, pipeline in best_models.items():
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')  # ou 'macro'/'weighted' pour multiclasse
        recall = recall_score(y_test, y_pred, average='binary')

        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Recall': recall
        })

    return pd.DataFrame(results)

def plot_model_performance(results_df):
    results_df.set_index('Model', inplace=True)
    results_df.plot(kind='bar', figsize=(10, 6))
    plt.title("Model Performance on Test Set")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Évaluation des modèles sur l'ensemble de test et affichage des résultats
test_results = evaluate_on_test_set(best_models, X_test, y_test)
print(test_results)

# Tracé des performances des modèles
plot_model_performance(test_results)
