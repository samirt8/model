import streamlit as st
import requests

# Titre de l'application
st.title('Application de Scoring Client')

# URL de l'API Flask déployée (à modifier avec l'URL réelle si déployée sur un serveur distant)
api_url = 'http://127.0.0.1:5000/predict'

# Champ de saisie pour l'ID du client
client_id = st.text_input('Entrez l\'ID du client', '')

# Bouton pour déclencher la prédiction
if st.button('Faire la prédiction'):
    if client_id:  # Vérifie que l'ID du client n'est pas vide
        # Effectue une requête POST à l'API avec l'ID du client
        try:
            response = requests.post(api_url, json={'id_client': client_id})
            
            # Affiche le contenu brut de la réponse pour le débogage
            st.write("Code de statut HTTP:", response.status_code)
            st.write("Réponse brute:", response.text)
            
            if response.status_code == 200:  # Vérifie si la requête a réussi
                result = response.json()
                # Affiche les résultats de la prédiction
                st.write(f"Probabilité de défaut : {result['probability']:.2%}")
                st.write(f"Prédiction : {result['prediction']}")
            else:
                error_message = response.json().get('error', 'Erreur inconnue')
                st.write(f"Erreur : {error_message}")
        except requests.exceptions.RequestException as e:
            st.write(f"Erreur lors de la connexion à l'API : {e}")
    else:
        st.write("Veuillez entrer un ID de client valide.")
