diff --git a/README.md b/README.md
index 4da581f4f828ae4a663e3ad9191862d3257b5600..c37bea3e8a18d121a2a85d75376ef56f537bf9a7 100644
--- a/README.md
+++ b/README.md
@@ -1,57 +1,99 @@
-# BigDataSupplyChain
-E-commerce Optimization Using Big Data and Artificial Intelligence
-🔍 Project Objective
-This project aims to enhance the efficiency of the supply chain and improve customer experience for an e-commerce platform by leveraging Big Data technologies and predictive Artificial Intelligence. The solution is based on a rich e-commerce dataset containing order information, delivery times, customer behavior, purchase history, and logistics metadata.
+# BigdataSupplyChain Pro
 
-⚙️ Technologies Used
-Apache Kafka: Real-time ingestion of data from the e-commerce platform (orders, payments, delivery tracking).
+> **Plateforme intelligente de pilotage supply chain e-commerce** : ingestion temps réel, IA prédictive et visualisation décisionnelle pour réduire les retards, optimiser les coûts et améliorer l’expérience client.
 
-Apache Spark: Distributed data processing for analytics, data transformation, and model training.
+---
 
-Python: Core language for implementing data pipelines and AI/ML models.
+## 🚀 Pourquoi ce projet ?
 
-Dash (Plotly): Interactive web application to visualize predictions, delivery performance, and customer segmentation.
+Les opérations e-commerce souffrent souvent de trois problèmes majeurs :
+- des prévisions de livraison peu fiables,
+- des retards détectés trop tard,
+- une connaissance client insuffisante pour agir rapidement.
 
-🧠 Integrated AI Modules
-1. 🔮 Delivery Time Prediction
-Model: Deep Learning – Regression (Dense Neural Network)
+**BigdataSupplyChain Pro** répond à ces enjeux avec une architecture orientée données massives et intelligence artificielle, conçue pour les équipes **logistique, service client et marketing**.
 
-Goal: Accurately predict the exact delivery time of an order based on factors such as product type, shipping method, location, carrier history, and weather conditions.
+---
 
-Input Data: Order ID, order date, postal code, etc
+## 🎯 Impact business (version Pro)
 
-Output: Estimated delivery time (in days/hours).
+- **Réduction des livraisons en retard** grâce à un scoring de risque en amont.
+- **Amélioration de la promesse client** via une estimation plus précise des délais.
+- **Priorisation opérationnelle** des commandes à risque avant incident.
+- **Segmentation client actionnable** pour campagnes marketing ciblées.
+- **Pilotage en temps réel** via dashboard interactif pour la prise de décision.
 
-2. ⏰ Late Delivery Prediction
-Model: Deep Learning – Binary Regression (Sigmoid classifier)
+---
 
-Goal: Predict whether an order is likely to be delivered late compared to the promised timeframe.
+## 🧱 Architecture fonctionnelle
 
-Output: Probability of delay; high-risk orders can be flagged for early intervention.
+1. **Ingestion temps réel (Kafka)**
+   - Flux commandes, paiements, tracking transporteurs et événements logistiques.
+2. **Traitement distribué (Spark)**
+   - Nettoyage, transformation, feature engineering et préparation des datasets.
+3. **Modélisation IA (Python / Deep Learning + ML)**
+   - Prédiction du délai de livraison.
+   - Classification du risque de retard.
+   - Segmentation clients par clustering.
+4. **Restitution métier (Dash/Plotly)**
+   - Visualisation des KPI, alertes retard, clusters clients et filtres dynamiques.
 
-3. 👥 Customer Segmentation
-Model: Unsupervised Learning – K-Means Clustering
+---
 
-Goal: Identify customer groups based on behavior (purchase frequency, basket size, loyalty, etc.).
+## 🧠 Modules IA intégrés
 
-Use Case: Personalized marketing, loyalty programs, customer targeting.
+### 1) Prédiction du délai de livraison
+- **Type** : Régression (réseau de neurones dense)
+- **Objectif** : Estimer le délai de livraison attendu (jours/heures)
+- **Variables exploitées** : type produit, mode d’expédition, zone géographique, historique transporteur, temporalité, etc.
 
-📊 Interactive Dashboard (Dash)
-A Dash application is built to provide insights to logistics and marketing teams:
+### 2) Prédiction de retard
+- **Type** : Classification binaire (sortie sigmoïde)
+- **Objectif** : Calculer la probabilité qu’une commande soit livrée en retard
+- **Valeur métier** : Permet d’anticiper les actions correctives (priorisation, communication proactive, reroutage)
 
-Delivery Forecast Dashboard: View predicted delivery times and delay risks by region, carrier, or product type.
+### 3) Segmentation client
+- **Type** : Clustering non supervisé (K-Means)
+- **Objectif** : Regrouper les clients par comportements d’achat
+- **Cas d’usage** : campagnes personnalisées, fidélisation, ciblage à forte valeur
 
-Dynamic Customer Segmentation: Visualize clusters and explore customer profiles.
+---
 
-Real-time Filters: Filter by time, geography, and product category.
+## 📊 Dashboard décisionnel
 
-📦 Expected Benefits
-More accurate delivery time estimations
+L’application Dash permet de :
+- suivre les délais prévus vs retards potentiels,
+- filtrer par région, transporteur, période et catégorie produit,
+- analyser les segments clients et leur contribution business,
+- faciliter la prise de décision inter-équipes (Ops, CX, Marketing).
 
-Reduction in late deliveries
+---
 
-Early detection of logistics issues
+## 🛠️ Stack technologique
 
-Targeted marketing through smart segmentation
+- **Apache Kafka** — streaming et ingestion événementielle
+- **Apache Spark** — traitement big data et pipelines distribués
+- **Python** — orchestration data, ML/DL et scripts opérationnels
+- **Dash / Plotly** — visual analytics et interface métier
+
+---
+
+## 📁 Composants principaux du repository
+
+- `kafka_data.py`, `kafka_data_logs.py`, `data_recept.py`, `data_recept_logs.py` : ingestion et réception de flux
+- `pipeline_for_model_day/` : pipeline de prédiction du délai
+- `pipline/` : transformations de données (cast/encodage)
+- `pipeline_segm/` : logique de segmentation
+- `prediction_data.py`, `prediction_data_segment.py` : scripts d’inférence
+- `dashboard.py` : application de visualisation
+
+---
+
+## ✅ Proposition de valeur
+
+**BigdataSupplyChain Pro** transforme les données opérationnelles en décisions concrètes :
+- moins de retards,
+- plus de fiabilité,
+- meilleure satisfaction client,
+- et un pilotage supply chain orienté performance.
 
-Improved customer satisfaction and retention
