import streamlit as st
import pandas as pd
import joblib

# Configuration de la page web
st.set_page_config(page_title="Prédiction Risque Cancer", page_icon="🩺", layout="wide")

# Chargement du modèle et des colonnes sauvegardés
@st.cache_resource
def charger_modele():
    modele = joblib.load('modele_cancer.pkl')
    colonnes = joblib.load('colonnes_cancer.pkl')
    return modele, colonnes

clf, colonnes = charger_modele()

# En-tête de l'application
st.title("🩺 Prédiction du niveau de risque du cancer du poumon")
st.write("Renseignez les informations et symptômes du patient ci-dessous (échelle de 1 à 9 pour la majorité des critères).")
st.divider()

# Création d'un formulaire organisé sur 3 colonnes
with st.form("formulaire_patient"):
    col1, col2, col3 = st.columns(3)
    entrees = {}

    with col1:
        st.subheader("Profil & Environnement")
        entrees['Age'] = st.number_input("Âge", min_value=1, max_value=120, value=35)
        genre = st.selectbox("Sexe", options=["Homme (1)", "Femme (2)"])
        entrees['Gender'] = 1 if "1" in genre else 2
        entrees['Air Pollution'] = st.slider("Pollution de l'air", 1, 9, 3)
        entrees['Alcohol use'] = st.slider("Consommation d'alcool", 1, 9, 3)
        entrees['Dust Allergy'] = st.slider("Allergie à la poussière", 1, 9, 3)
        entrees['OccuPational Hazards'] = st.slider("Risques professionnels", 1, 9, 3)
        entrees['Genetic Risk'] = st.slider("Risque génétique", 1, 9, 3)
        entrees['chronic Lung Disease'] = st.slider("Maladie pulmonaire chronique", 1, 9, 3)

    with col2:
        st.subheader("Mode de vie & Habitudes")
        entrees['Balanced Diet'] = st.slider("Alimentation équilibrée", 1, 9, 3)
        entrees['Obesity'] = st.slider("Obésité", 1, 9, 3)
        entrees['Smoking'] = st.slider("Tabagisme actif", 1, 9, 3)
        entrees['Passive Smoker'] = st.slider("Tabagisme passif", 1, 9, 3)
        entrees['Snoring'] = st.slider("Ronflements", 1, 9, 3)
        entrees['Fatigue'] = st.slider("Fatigue", 1, 9, 3)
        entrees['Weight Loss'] = st.slider("Perte de poids", 1, 9, 3)

    with col3:
        st.subheader("Symptômes cliniques")
        entrees['Chest Pain'] = st.slider("Douleur thoracique", 1, 9, 3)
        entrees['Coughing of Blood'] = st.slider("Toux avec du sang", 1, 9, 3)
        entrees['Shortness of Breath'] = st.slider("Essoufflement", 1, 9, 3)
        entrees['Wheezing'] = st.slider("Respiration sifflante", 1, 9, 3)
        entrees['Swallowing Difficulty'] = st.slider("Difficulté à avaler", 1, 9, 3)
        entrees['Clubbing of Finger Nails'] = st.slider("Hippocratisme digital (ongles)", 1, 9, 3)
        entrees['Frequent Cold'] = st.slider("Rhumes fréquents", 1, 9, 3)
        entrees['Dry Cough'] = st.slider("Toux sèche", 1, 9, 3)

    # Bouton de validation
    bouton_predire = st.form_submit_button("🔍 Analyser le risque", use_container_width=True)

# Traitement et affichage du résultat
if bouton_predire:
    # Création du DataFrame dans l'ordre exact des colonnes d'entraînement
    df_patient = pd.DataFrame([entrees])[colonnes]
    
    prediction = clf.predict(df_patient)[0]
    probabilites = clf.predict_proba(df_patient)[0]
    confiance = max(probabilites) * 100

    st.divider()
    st.subheader("Résultat de l'analyse :")

    if prediction == "Low":
        st.success(f"🟢 **Niveau de risque : FAIBLE (Low)** — Confiance du modèle : {confiance:.1f} %")
    elif prediction == "Medium":
        st.warning(f"🟠 **Niveau de risque : MOYEN (Medium)** — Confiance du modèle : {confiance:.1f} %")
    else:
        st.error(f"🔴 **Niveau de risque : ÉLEVÉ (High)** — Confiance du modèle : {confiance:.1f} %")
