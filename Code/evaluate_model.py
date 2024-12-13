from sklearn.metrics import  classification_report, roc_auc_score
def evaluate_model(model,  X_test, y_test, model_type = "NN", dataset_balance = "balanced" ):
    print(f"Evaluation for {dataset_balance} dataset and model {model_type}:")

    #if random_forest
    if(model_type == "rf"):
        y_pred_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_probs = model.predict(X_test)

    y_pred = (y_pred_probs > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Apnea/Hypopnea"]))

    r_auc_score = roc_auc_score(y_test, y_pred_probs)
    print(f"ROC AUC Score: {r_auc_score:.4f}")


