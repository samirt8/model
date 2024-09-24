from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import lightgbm
import sklearn


app = Flask(__name__)

# Chemin vers le fichier lgbm_model.pkl
model_path = 'model/lgbm_model.pkl'
# Chargement du modèle
model = joblib.load(model_path)



# Charger les données d'entraînement
df = pd.read_csv(os.path.join('model', 'test_data.csv'))

df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)  # S'assurer que les IDs sont des entiers

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    all_id_client = list(df['SK_ID_CURR'].unique())
    seuil = 0.625

    data = request.form
    ID = data.get('id_client', '')

    # Débogage - afficher l'ID envoyé par l'utilisateur
    print(f"ID envoyé par l'utilisateur: {ID}")

    try:
        ID = int(ID)  # Utiliser int si l'ID dans test_data.csv est un entier
    except ValueError:
        return jsonify({"error": "ID invalide"}), 400

    # Vérifiez que l'ID est bien trouvé dans df
    if df[df['SK_ID_CURR'] == ID].empty:
        # Débogage - Affiche le type et le format des IDs
        print(f"Type de l'ID fourni: {type(ID)}")
        print(f"ID {ID} n'existe pas dans la base de données")
        print(f"Les 5 premiers IDs dans la base : {df['SK_ID_CURR'].head()}")  # Afficher les 5 premiers IDs
        return jsonify({"error": "Ce client n'est pas répertorié"}), 404
    else:
        print(f"ID {ID} trouvé dans les données.")

    X = df[df['SK_ID_CURR'] == ID].drop(['SK_ID_CURR'], axis=1)

    # Obtenez les colonnes attendues
    expected_columns = model.feature_name_

    # Vérifiez les colonnes manquantes ou en excès
    missing_cols = set(expected_columns) - set(X.columns)
    extra_cols = set(X.columns) - set(expected_columns)

    # Débogage - Afficher les colonnes manquantes et en trop
    print(f"Colonnes manquantes: {missing_cols}")
    print(f"Colonnes en trop: {extra_cols}")

    # Ajoutez les colonnes manquantes avec des valeurs par défaut
    for col in missing_cols:
        X[col] = 0

    # Réorganisez les colonnes pour correspondre à l'ordre attendu
    X = X[expected_columns]

    if X.shape[1] != model.n_features_in_:
        return jsonify({
            "error": "Nombre de caractéristiques incorrect",
            "expected_features_count": model.n_features_in_,
            "received_features_count": X.shape[1]
        }), 400

    try:
        probability_default_payment = model.predict_proba(X)[:, 1][0]
    except Exception as e:
        return jsonify({'error': f'Erreur lors de la prédiction: {str(e)}'}), 500

    prediction = "Prêt NON Accordé, risque de défaut" if probability_default_payment >= seuil else "Prêt Accordé"

    return jsonify({"probability": probability_default_payment, "prediction": prediction})


@app.route('/prediction_complete')
def pred_model():
    try:
        Xtot = df.drop(['SK_ID_CURR'], axis=1)
        seuil = 0.625
        y_pred = model.predict_proba(Xtot)[:, 1]
        y_seuil = y_pred >= seuil
        y_seuil = np.array(y_seuil > 0) * 1

        df_pred = df.copy()
        df_pred['Proba'] = y_pred
        df_pred['PREDICTION'] = y_seuil

        return df_pred.to_json(orient='index')

    except Exception as e:
        return jsonify({'error': 'Erreur lors de la génération des prédictions'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
