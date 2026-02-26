# 🚚 BigdataSupplyChain Pro

> **AI-Powered Supply Chain Intelligence Platform for E-commerce**  
> Ingestion temps réel, IA prédictive et dashboard décisionnel pour réduire les retards, optimiser les coûts et améliorer l’expérience client.

---

## 🔍 Project Overview

BigdataSupplyChain Pro est une plateforme Big Data & Intelligence Artificielle conçue pour optimiser la performance supply chain des plateformes e-commerce.

Elle centralise les flux logistiques et commerciaux afin de fournir une vue **unifiée, prédictive et actionnable** des opérations.

🎯 Public cible :
- Équipes **Opérations / Logistique**
- Équipes **Service Client (CX)**
- Équipes **Marketing & CRM**

Objectif : passer d’un pilotage réactif à un pilotage **proactif basé sur la donnée**.

---

## 🚀 Business Impact

- 📉 Réduction des livraisons en retard via un scoring de risque en amont
- 📦 Amélioration de la précision des délais estimés
- ⚡ Priorisation opérationnelle des commandes à risque
- 🎯 Segmentation client actionnable pour marketing ciblé
- 📊 Pilotage en temps réel grâce à un dashboard interactif

---

## 🧱 Architecture Globale

### 1️⃣ Ingestion Temps Réel
- **Apache Kafka**
  - Flux commandes
  - Paiements
  - Tracking transporteurs
  - Logs opérationnels

### 2️⃣ Traitement Distribué
- **Apache Spark**
  - Nettoyage & transformation
  - Feature engineering
  - Préparation des datasets

### 3️⃣ Modélisation IA
- Prédiction du délai de livraison
- Classification du risque de retard
- Segmentation client

### 4️⃣ Restitution Métier
- **Dash / Plotly**
  - Dashboard décisionnel interactif
  - Visualisation KPI et alertes

---

## 🧠 Modules IA

### 🔮 1. Prédiction du délai de livraison
- **Type** : Régression (Deep Learning – Dense Neural Network)
- **Output** : Délai estimé (jours/heures)
- **Valeur métier** : Fiabiliser la promesse client

---

### ⏰ 2. Prédiction de retard
- **Type** : Classification binaire (Sigmoid)
- **Output** : Probabilité de retard
- **Valeur métier** : Déclencher des actions correctives avant incident

---

### 👥 3. Segmentation client
- **Type** : Clustering (K-Means)
- **Output** : Groupes clients homogènes
- **Usage** :
  - Personnalisation marketing
  - Fidélisation
  - Priorisation valeur client

---

## 📊 Dashboard Décisionnel

Le dashboard permet de :

- Suivre délais prévus vs réels
- Identifier les commandes à haut risque
- Filtrer par région, transporteur, période, catégorie
- Visualiser la performance par segment client
- Aligner Ops, CX et Marketing autour des mêmes KPI

---

## 📈 KPI Recommandés

- Taux de commandes en retard
- Écart moyen délai prédit vs réel
- Part de commandes à risque élevé
- Temps moyen de résolution incident
- Performance par segment client

---

## 🛠️ Stack Technologique

- **Apache Kafka** — Streaming événementiel
- **Apache Spark** — Traitement Big Data distribué
- **Python** — Pipelines data & ML
- **Deep Learning / ML**
- **Dash / Plotly** — Visualisation interactive

---

## 📁 Structure du Repository

### Ingestion
- `kafka_data.py`
- `kafka_data_logs.py`
- `data_recept.py`
- `data_recept_logs.py`

### Pipelines IA
- `pipeline_for_model_day/`
- `pipeline_segm/`
- `pipline/`

### Inférence & Application
- `prediction_data.py`
- `prediction_data_segment.py`
- `dashboard.py`

---

## ⚡ Quick Start

1. Installer les dépendances Python
2. Lancer Kafka
3. Démarrer les scripts d’ingestion
4. Exécuter les pipelines ML
5. Lancer le dashboard

---

## 💎 Proposition de Valeur

BigdataSupplyChain Pro transforme les données opérationnelles en décisions stratégiques :

- ✅ Moins de retards
- ✅ Meilleure précision
- ✅ Décisions plus rapides
- ✅ Expérience client améliorée
- ✅ Supply chain plus rentable
