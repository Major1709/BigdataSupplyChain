# BigdataSupplyChain Pro

Plateforme Data & IA pour optimiser la performance supply chain e-commerce, réduire les retards de livraison et améliorer l’expérience client.

## Sommaire
- [1. Vision produit](#1-vision-produit)
- [2. Problèmes adressés](#2-problèmes-adressés)
- [3. Impact business attendu](#3-impact-business-attendu)
- [4. Architecture globale](#4-architecture-globale)
- [5. Modules IA](#5-modules-ia)
- [6. Dashboard décisionnel](#6-dashboard-décisionnel)
- [7. Stack technologique](#7-stack-technologique)
- [8. Organisation du repository](#8-organisation-du-repository)
- [9. Mise en route (quick start)](#9-mise-en-route-quick-start)
- [10. KPI de pilotage recommandés](#10-kpi-de-pilotage-recommandés)
- [11. Proposition de valeur](#11-proposition-de-valeur)

---

## 1. Vision produit

**BigdataSupplyChain Pro** centralise les flux logistiques et commerciaux pour fournir une vue unifiée, prédictive et actionnable des opérations.

Le projet cible prioritairement :
- les équipes **Opérations / Logistique**,
- les équipes **Service Client (CX)**,
- les équipes **Marketing & CRM**.

Objectif : passer d’un pilotage réactif à un pilotage **proactif**, basé sur des modèles de prédiction et des indicateurs temps réel.

---

## 2. Problèmes adressés

Les plateformes e-commerce rencontrent fréquemment :
- des estimations de délai imprécises,
- une détection tardive des commandes à risque,
- un manque de segmentation client opérationnelle,
- une difficulté à aligner les équipes autour des mêmes KPI.

BigdataSupplyChain Pro répond à ces limites via une chaîne complète : ingestion, traitement big data, IA, visualisation.

---

## 3. Impact business attendu

- **Réduire le taux de retard** en identifiant les commandes à risque plus tôt.
- **Améliorer la fiabilité de la promesse de livraison** grâce à la prédiction des délais.
- **Prioriser les interventions opérationnelles** (escalade transporteur, reroutage, communication proactive).
- **Activer un marketing plus précis** via la segmentation comportementale.
- **Augmenter la satisfaction client** grâce à une meilleure transparence et moins d’incidents.

---

## 4. Architecture globale

### 4.1 Ingestion temps réel
- **Kafka** collecte les événements : commandes, paiements, suivi transport, logs opérationnels.

### 4.2 Traitement distribué
- **Spark** réalise le nettoyage, le typage, les enrichissements et la préparation des features.

### 4.3 Modélisation IA
- Modèles de prédiction de délai.
- Modèles de classification du risque de retard.
- Modèles de segmentation client.

### 4.4 Restitution métier
- **Dash/Plotly** fournit une interface de pilotage pour les équipes métiers.

---

## 5. Modules IA

### 5.1 Prédiction du délai de livraison
- **Type** : Régression (réseau de neurones dense)
- **Sortie** : délai estimé (jours/heures)
- **Usage** : fiabiliser la promesse client et planifier les opérations.

### 5.2 Prédiction de retard
- **Type** : Classification binaire (sigmoïde)
- **Sortie** : probabilité de retard par commande
- **Usage** : déclencher des actions correctives avant incident.

### 5.3 Segmentation client
- **Type** : Clustering non supervisé (K-Means)
- **Sortie** : groupes clients à comportements homogènes
- **Usage** : personnalisation marketing, fidélisation, priorisation de la valeur client.

---

## 6. Dashboard décisionnel

Le dashboard permet de :
- suivre les délais prévus et le risque de retard,
- filtrer par région, période, transporteur, catégorie produit,
- visualiser les segments clients et leur performance,
- partager une lecture commune des KPI entre Ops, CX et Marketing.

---

## 7. Stack technologique

- **Apache Kafka** : ingestion événementielle en continu
- **Apache Spark** : pipelines de traitement et transformations big data
- **Python** : scripts data, entraînement et inférence des modèles
- **Dash / Plotly** : application analytique interactive

---

## 8. Organisation du repository

### Flux et ingestion
- `kafka_data.py`
- `kafka_data_logs.py`
- `data_recept.py`
- `data_recept_logs.py`

### Pipelines ML / IA
- `pipeline_for_model_day/` : prédiction du délai de livraison
- `pipeline_segm/` : segmentation client
- `pipline/` : transformations de colonnes / preprocessing

### Inférence et application
- `prediction_data.py`
- `prediction_data_segment.py`
- `dashboard.py`

---

## 9. Mise en route (quick start)

1. Installer les dépendances Python du projet.
2. Démarrer les services de streaming (Kafka).
3. Lancer les scripts d’ingestion.
4. Exécuter les pipelines de préparation / entraînement.
5. Démarrer le dashboard pour visualiser les résultats.

> Remarque : adaptez les paramètres (brokers Kafka, chemins de données, environnement Python) selon votre infrastructure.

---

## 10. KPI de pilotage recommandés

- Taux de commandes livrées en retard
- Écart moyen entre délai prédit et délai réel
- Part de commandes classées “à risque élevé”
- Temps moyen de résolution des incidents logistiques
- Performance par segment client (fréquence, panier, rétention)

---

## 11. Proposition de valeur

**BigdataSupplyChain Pro** transforme les données opérationnelles en leviers de performance mesurables :
- moins de retards,
- meilleure précision des prévisions,
- décisions plus rapides,
- meilleure expérience client,
- pilotage supply chain plus rentable et plus fiable.
