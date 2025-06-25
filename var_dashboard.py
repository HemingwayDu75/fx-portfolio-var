import pandas as pd
import numpy as np
from scipy.stats import norm
import tkinter as tk
from tkinter import filedialog
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # === Paramètres généraux ===

    # Niveau de confiance (par exemple, 95% ou 99%)
    confidence_level = 0.95

    # Horizon temporel en jours
    time_horizon = 1

    # Calcul du z-score correspondant au niveau de confiance
    z_score = norm.ppf(confidence_level)

    # === Définition manuelle des devises et taux sans risque ===

    # Devise de risque (devise que vous achetez ou vendez)
    risk_currency = 'USD'

    # Devise fonctionnelle (votre devise comptable)
    functional_currency = 'EUR'

    # Taux d'intérêt sans risque (en décimal)
    risk_free_rate_risk_currency = 0.05  # Taux sans risque de la devise de risque (par exemple, USD)
    risk_free_rate_functional_currency = 0.0335  # Taux sans risque de la devise fonctionnelle (par exemple, EUR)

    # === Sélection des fichiers via une interface graphique ===

    # Initialiser l'interface tkinter
    root = tk.Tk()
    root.withdraw()  # Cacher la fenêtre principale

    # Demander à l'utilisateur de sélectionner le fichier 'SPOT'
    spot_file_path = filedialog.askopenfilename(title="Sélectionnez le fichier SPOT (cours historiques)",
                                                filetypes=[("Fichiers texte", "*.txt *.csv"), ("Tous les fichiers", "*.*")])

    if not spot_file_path:
        print("Aucun fichier SPOT sélectionné. Le programme va se terminer.")
        sys.exit()

    # Demander à l'utilisateur de sélectionner le fichier du portefeuille
    portfolio_file_path = filedialog.askopenfilename(title="Sélectionnez le fichier du portefeuille",
                                                     filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")])

    if not portfolio_file_path:
        print("Aucun fichier portefeuille sélectionné. Le programme va se terminer.")
        sys.exit()

    # === Lecture et traitement des données du fichier SPOT ===

    # Déterminer le délimiteur en fonction de l'extension du fichier
    if spot_file_path.endswith('.csv'):
        delimiter = ','
    else:
        delimiter = '\t'  # Ajustez selon le format réel de votre fichier

    try:
        # Lire le fichier SPOT
        spot_data = pd.read_csv(spot_file_path, delimiter=delimiter)
        print("Fichier SPOT chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier SPOT : {e}")
        sys.exit()

    # Convertir la colonne 'date' en format datetime
    spot_data['date'] = pd.to_datetime(spot_data['date'], dayfirst=True, errors='coerce')

    # Supprimer les lignes avec des dates non valides
    spot_data.dropna(subset=['date'], inplace=True)

    # Convertir la colonne 'clot' en numérique
    spot_data['clot'] = pd.to_numeric(spot_data['clot'], errors='coerce')

    # Supprimer les lignes avec des taux de change non valides
    spot_data.dropna(subset=['clot'], inplace=True)

    # Calculer les rendements logarithmiques quotidiens
    spot_data.sort_values('date', inplace=True)
    spot_data['Log_Returns'] = np.log(spot_data['clot'] / spot_data['clot'].shift(1))

    # Supprimer les valeurs NaN résultant du décalage
    spot_data.dropna(subset=['Log_Returns'], inplace=True)

    # === Calculer la volatilité quotidienne du taux de change avec pondération exponentielle ===
    
    # Définir le facteur de lissage lambda (plus il est proche de 1, plus les données anciennes ont du poids)
    lambda_factor = 0.97
    
    # Calcul de la volatilité pondérée exponentiellement (EWMA)
    spot_data['EWMA_Volatility'] = spot_data['Log_Returns'].ewm(span=(2 / (1 - lambda_factor) - 1), adjust=False).std()
    
    # Extraire la volatilité la plus récente
    volatility_daily = spot_data['EWMA_Volatility'].iloc[-1]

    # Récupérer le taux spot actuel
    current_spot_rate = spot_data['clot'].iloc[-1]

    # === Lecture et traitement des données du portefeuille ===

    # Lecture du fichier portefeuille avec l'encodage latin-1
    try:
        portfolio = pd.read_csv(portfolio_file_path, encoding='latin-1', sep=';')
        print("Fichier portefeuille chargé avec succès avec l'encodage latin-1.")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier portefeuille : {e}")
        sys.exit()

    # Convertir les dates en format datetime
    portfolio['Trade Date'] = pd.to_datetime(portfolio['Trade Date'], dayfirst=True, errors='coerce')
    portfolio['ValueDate'] = pd.to_datetime(portfolio['ValueDate'], dayfirst=True, errors='coerce')

    # Calculer le temps jusqu'à l'échéance en années
    portfolio['Time_to_Maturity'] = (portfolio['ValueDate'] - pd.to_datetime('today')).dt.days / 365.25

    # Gérer les positions avec une échéance passée
    portfolio.loc[portfolio['Time_to_Maturity'] < 0, 'Time_to_Maturity'] = 0

    # Convertir les colonnes numériques en float
    numeric_cols = ['Amount1', 'Fwd/Strike']
    for col in numeric_cols:
        portfolio[col] = pd.to_numeric(portfolio[col], errors='coerce')

    # Ajouter les taux spot et volatilité aux positions
    portfolio['Spot Rate'] = current_spot_rate
    portfolio['Implied_Volatility'] = volatility_daily

    # Initialiser une colonne pour l'exposition delta en devise de risque
    portfolio['Position_Delta'] = 0.0

    # Définir les fonctions pour calculer le delta
    def calculate_option_delta(row):
        S = row['Spot Rate']
        K = row['Fwd/Strike']
        T = row['Time_to_Maturity']
        sigma = row['Implied_Volatility']
        r_d = risk_free_rate_functional_currency
        r_f = risk_free_rate_risk_currency
        option_type = row['Type']

        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0

        try:
            d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        except Exception:
            return 0

        if option_type == 'Call':
            delta = np.exp(-r_f * T) * norm.cdf(d1)
        elif option_type == 'Put':
            delta = -np.exp(-r_f * T) * norm.cdf(-d1)
        else:
            delta = 0
        return delta

    # Calculer le delta de chaque position
    for idx, row in portfolio.iterrows():
        amount = row['Amount1']
        direction = row['Direction']
        option_type = row['Type']
        delta = 0

        if option_type in ['Call', 'Put']:
            delta = calculate_option_delta(row)

            # Ajuster le delta selon la direction
            if direction in ['Achat', 'Buy']:
                position_delta = delta * amount
            elif direction in ['Vente', 'Sell']:
                position_delta = -delta * amount
            else:
                print(f"Direction inconnue pour la position {idx}: {direction}. Delta mis à zéro.")
                position_delta = 0
        else:
            # Pour les forwards ou termes
            if direction in ['Achat', 'Buy']:
                position_delta = amount
            elif direction in ['Vente', 'Sell']:
                position_delta = -amount
            else:
                print(f"Direction inconnue pour la position {idx}: {direction}. Delta mis à zéro.")
                position_delta = 0

        # Enregistrer le delta de la position en devise de risque
        portfolio.at[idx, 'Position_Delta'] = position_delta

    # Exposition totale delta-équivalente du portefeuille en devise de risque
    total_delta_exposure = portfolio['Position_Delta'].sum()

    # Volatilité du portefeuille en devise de risque
    portfolio_volatility = abs(total_delta_exposure) * volatility_daily

    # Calcul de la VaR en devise de risque
    VaR_risk_currency = z_score * portfolio_volatility * np.sqrt(time_horizon)

    # Conversion de la VaR en devise fonctionnelle (division par le taux spot)
    VaR_functional_currency = VaR_risk_currency / current_spot_rate

    # === Calcul de la VaR pour chaque position ===

    # Calculer la VaR de chaque position en devise de risque
    portfolio['VaR_risk_currency'] = z_score * abs(portfolio['Position_Delta']) * volatility_daily * np.sqrt(time_horizon)

    # Conversion de la VaR en devise fonctionnelle
    portfolio['VaR_functional_currency'] = portfolio['VaR_risk_currency'] / current_spot_rate

    # === Organisation des éléments de reporting dans un DataFrame pandas ===

    reporting_data = {
        'Élément': [
            'Volatilité quotidienne du taux de change',
            f'Exposition delta-équivalente du portefeuille (en {risk_currency})',
            f'Volatilité du portefeuille (en {risk_currency})',
            f'VaR à {confidence_level*100:.0f}% sur {time_horizon} jour(s) (en {risk_currency})',
            f'VaR à {confidence_level*100:.0f}% sur {time_horizon} jour(s) (en {functional_currency})'
        ],
        'Valeur': [
            f"{volatility_daily:.5f}",
            f"{total_delta_exposure:.2f}",
            f"{portfolio_volatility:.2f}",
            f"{VaR_risk_currency:.2f}",
            f"{VaR_functional_currency:.2f}"
        ]
    }

    report_df = pd.DataFrame(reporting_data)

    # Afficher le reporting global
    print("\n=== Rapport de calcul de la VaR ===")
    print(report_df.to_string(index=False))

    # Afficher le détail des positions avec delta et VaR
    reporting_columns = ['Type', 'Direction', 'Amount1', 'Ccy1', 'Ccy2', 'Position_Delta', 'VaR_risk_currency', 'VaR_functional_currency']
    portfolio_report = portfolio[reporting_columns].copy()

    # Arrondir les colonnes numériques pour une meilleure lisibilité
    portfolio_report['Amount1'] = portfolio_report['Amount1'].map('{:,.2f}'.format)
    portfolio_report['Position_Delta'] = portfolio_report['Position_Delta'].map('{:,.2f}'.format)
    portfolio_report['VaR_risk_currency'] = portfolio_report['VaR_risk_currency'].map('{:,.2f}'.format)
    portfolio_report['VaR_functional_currency'] = portfolio_report['VaR_functional_currency'].map('{:,.2f}'.format)

    print("\n=== Détails des positions avec delta et VaR ===")
    print(portfolio_report.to_string(index=False))

    # Afficher les 5 principales positions par exposition delta
    top_positions = portfolio[['Type', 'Direction', 'Amount1', 'Ccy1', 'Ccy2', 'Position_Delta']].copy()
    top_positions['Abs_Position_Delta'] = top_positions['Position_Delta'].abs()
    top_positions.sort_values('Abs_Position_Delta', ascending=False, inplace=True)
    top_positions = top_positions.head(5)

    print("\n=== Top 5 des positions par exposition delta ===")
    print(top_positions[['Type', 'Direction', 'Amount1', 'Ccy1', 'Ccy2', 'Position_Delta']].to_string(index=False))

if __name__ == "__main__":
    main()
