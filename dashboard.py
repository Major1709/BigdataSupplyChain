# DASH VERSION
import dash
from dash import dcc, html, Input, Output,dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour,desc
from pyspark.sql.types import IntegerType, DoubleType
import pycountry
from pyspark.ml.linalg import DenseVector, SparseVector
import numpy as np
import time
import threading

# Initialiser Spark
spark = SparkSession.builder.appName("DashFig").config("spark.ui.port", "4046").getOrCreate()

# Lire les fichiers HDFS
df_order = spark.read.option("header", "true").csv("hdfs://localhost:9000/projet/gold/Order")
df_transaction = spark.read.option("header", "true").csv("hdfs://localhost:9000/projet/gold/Transaction")
df_orderitem = spark.read.option("header", "true").csv("hdfs://localhost:9000/projet/gold/OrderItem")
df_product = spark.read.option("header", "true").csv("hdfs://localhost:9000/projet/gold/Product")
df_customer = spark.read.option("header", "true").csv("hdfs://localhost:9000/projet/gold/Customer")
df_delivery = spark.read.option("header", "true").csv("hdfs://localhost:9000/projet/gold/Delivery")
df_department = spark.read.option("header", "true").csv("hdfs://localhost:9000/projet/gold/Department")
df_category = spark.read.option("header", "true").csv("hdfs://localhost:9000/projet/gold/Category")
df_logs = spark.read.option("header", "true").csv("hdfs://localhost:9000/projet/raw/logs/tokenized_access_logs.csv")

# Convertir la colonne Date en timestamp
df_logs = df_logs.withColumn("Date", to_timestamp("Date", "M/d/yyyy H:mm"))

# Extraire l'heure
df_logs = df_logs.withColumn("Hour", hour("Date"))

df_transaction = df_transaction \
    .withColumn("days_for_shipping_(real)", col("days_for_shipping_(real)").cast(IntegerType())) \
    .withColumn("days_for_shipment_(scheduled)", col("days_for_shipment_(scheduled)").cast(IntegerType())) \
    .withColumn("benefit_per_order", col("benefit_per_order").cast(DoubleType())) \
    .withColumn("sales_per_customer", col("sales_per_customer").cast(DoubleType())) \
    .withColumn("late_delivery_risk", col("Late_delivery_risk").cast(IntegerType()))

df_customer = df_customer \
    .withColumn("latitude", col("latitude").cast(DoubleType())) \
    .withColumn("longitude", col("longitude").cast(DoubleType())) 

df_product = df_product \
    .withColumn("product_price", col("product_price").cast(DoubleType())) 

df_orderitem = df_orderitem \
    .withColumn("order_item_quantity", col("order_item_quantity").cast(IntegerType())) \
    .withColumn("order_item_discount", col("order_item_discount").cast(DoubleType())) \
    .withColumn("order_item_discount_rate", col("order_item_discount_rate").cast(DoubleType())) \
    .withColumn("order_item_product_price", col("order_item_product_price").cast(DoubleType())) \
    .withColumn("order_item_profit_ratio", col("order_item_profit_ratio").cast(DoubleType())) \
    .withColumn("order_item_total", col("order_item_total").cast(DoubleType()))

df_delivery = df_delivery \
    .withColumn("days_for_shipping_(real)", col("days_for_shipping_(real)").cast(IntegerType())) \
    .withColumn("days_for_shipment_(scheduled)", col("days_for_shipment_(scheduled)").cast(IntegerType())) \
    .withColumn("late_delivery_risk", col("late_delivery_risk").cast(IntegerType())) \
    .withColumn("benefit_per_order", col("benefit_per_order").cast(DoubleType())) \
    .withColumn("order_profit_per_order", col("order_profit_per_order").cast(DoubleType()))

# Temp views
for name, df in zip(
    ["transaction", "order", "orderitem", "product", "customer", "delivery", "department", "category"],
    [df_transaction, df_order, df_orderitem, df_product, df_customer, df_delivery, df_department, df_category]
):
    df.createOrReplaceTempView(name)



# Initialisation de Dash
external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

color_p = "#1a2245"
color_s = "#1e2c58"

# Layout de l'application
app.layout = html.Div([

    dcc.Interval(id='interval-component', interval=15*1000, n_intervals=0),
    html.H1("Dashboard Supply Chain", style={'textAlign': 'left', 'margin-bottom': '50px','margin-left' : '20px', 'color': '#FFFFFF'}),

    dbc.Row([
        dbc.Col([
            html.H4("Total par statut de commande"),
            dcc.Graph(id='order-status')],style={'backgroundColor': color_s, 'border': '1px solid #052546', 'borderRadius': '20px', 'padding': '20px', 'margin':'15px','width': '45%'}),
        dbc.Col([
            html.H4("\U0001F4B0 Évolution des ventes dans le temps"),
            dcc.Graph(id='sales-time')],style={'backgroundColor': color_s, 'border': '1px solid #052546', 'borderRadius': '20px', 'padding': '20px', 'margin':'15px','width': '45%'}),                
    ], style={'margin': '20px'}),
    dbc.Row([
        dbc.Col([
            html.H4("Tableau de bord des retards de livraison"),
            dash_table.DataTable(
                id='data-table',
                columns=[],
                data=[],
                page_size=15,
                style_table={'overflowX': 'auto'},
                    style_cell={
                    'textAlign': 'left',
                    'backgroundColor': color_s,  # ✅ Fond des cellules
                    'color': '#000000'  # ✅ Couleur du texte
                },
                style_header={
                    'backgroundColor': '#052546',  # ✅ Fond de l'en-tête
                    'color': 'white',  # ✅ Texte en blanc
                    'fontWeight': 'bold'
                },
                style_data={
                    'backgroundColor': '#f9f9f9',  # ✅ Couleur de fond des lignes
                    'color': '#000000'  # ✅ Texte
                }
            
        )
        ],style={'backgroundColor': color_s, 'border': '1px solid #052546', 'borderRadius': '20px', 'padding': '20px', 'margin':'15px','width': '50%'}),
                       
    ], style={'margin': '20px'}),
    dbc.Row([
        dbc.Col([
            html.H4("Taux de retards de livraison global"),
            dcc.Graph(id='delivery-delay')],style={'backgroundColor': color_s, 'border': '1px solid #052546', 'borderRadius': '20px', 'padding': '20px', 'margin':'15px','width': '5%'}),
        
        dbc.Col([
            html.H4("Taux de retards de livraison à venir"),
            dcc.Graph(id='delivery-delay_real')],style={'backgroundColor': color_s, 'border': '1px solid #052546', 'borderRadius': '30px', 'padding': '20px', 'margin':'15px','width': '20%'})                 
    ], style={'margin': '20px'}),

    dbc.Row([
       dbc.Col([
            html.H4("Tableau de bord des Segmentations des clients pour la livraison"),
            dash_table.DataTable(
                id='data-table_segmentations',
                columns=[],
                data=[],
                page_size=15,
                style_table={'overflowX': 'auto'},
                    style_cell={
                    'textAlign': 'left',
                    'backgroundColor': color_s,  # ✅ Fond des cellules
                    'color': '#000000'  # ✅ Couleur du texte
                },
                style_header={
                    'backgroundColor': '#052546',  # ✅ Fond de l'en-tête
                    'color': 'white',  # ✅ Texte en blanc
                    'fontWeight': 'bold'
                },
                style_data={
                    'backgroundColor': '#f9f9f9',  # ✅ Couleur de fond des lignes
                    'color': '#000000'  # ✅ Texte
                }
            
        )
        ],style={'backgroundColor': color_s, 'border': '1px solid #052546', 'borderRadius': '20px', 'padding': '20px', 'margin':'15px','width': '50%'}),
        dbc.Col([
            html.H4("Taux des segmentations des clients pour la livraison"),
            dcc.Graph(id='segmentations_pie')],style={'backgroundColor': color_s, 'border': '1px solid #052546', 'borderRadius': '30px', 'padding': '20px', 'margin':'15px','width': '20%'})                 
    ], style={'margin': '20px'}),

    dbc.Row([
        dbc.Col([
            html.H4("Carte géographique des livraisons (par client)"),
            dcc.Graph(id='geo-map')],style={'backgroundColor': color_s, 'border': '1px solid #052546', 'borderRadius': '30px', 'padding': '20px', 'margin':'15px','width': '20%'}),
        dbc.Col([
            html.H4("Commandes par pays"),
            dcc.Graph(id='country-orders')],style={'backgroundColor': color_s, 'border': '1px solid #052546', 'borderRadius': '30px', 'padding': '20px', 'margin':'15px','width': '20%'})
    ], style={'margin': '20px'}),
    dbc.Row([
        dbc.Col([
            html.H4("\U0001F4E6 Top 10 des produits les plus vendus"),
            dcc.Graph(id='top-products')],style={'backgroundColor': color_s, 'border': '1px solid #052546', 'borderRadius': '30px', 'padding': '20px', 'margin':'15px','width': '20%'}),
        dbc.Col([
            html.H4("\U0001F4E6 Activité des client par heure chaque jours"),
            dcc.Graph(id='pic-products-heure')],style={'backgroundColor': color_s, 'border': '1px solid #052546', 'borderRadius': '30px', 'padding': '20px', 'margin':'15px','width': '20%'}),
    ], style={'margin': '20px'}),

    dbc.Row([
       dbc.Col([
            html.H4("Tableau de bord des Logs"),
            dash_table.DataTable(
                id='data-table-logs',
                columns=[],
                data=[],
                page_size=15,
                style_table={'overflowX': 'auto'},
                    style_cell={
                    'textAlign': 'left',
                    'backgroundColor': color_s,  # ✅ Fond des cellules
                    'color': '#000000'  # ✅ Couleur du texte
                },
                style_header={
                    'backgroundColor': '#052546',  # ✅ Fond de l'en-tête
                    'color': 'white',  # ✅ Texte en blanc
                    'fontWeight': 'bold'
                },
                style_data={
                    'backgroundColor': '#f9f9f9',  # ✅ Couleur de fond des lignes
                    'color': '#000000'  # ✅ Texte
                }
            
        )
        ],style={'backgroundColor': color_s, 'border': '1px solid #052546', 'borderRadius': '20px', 'padding': '20px', 'margin':'15px','width': '50%'}),
        dbc.Col([
            html.H4("Top des produit les plus visité"),
            dcc.Graph(id='top-product-visite')],style={'backgroundColor': color_s, 'border': '1px solid #052546', 'borderRadius': '30px', 'padding': '20px', 'margin':'15px','width': '20%'})                 
    ], style={'margin': '20px'}),

], style={'padding': '10px 55px', 'backgroundColor': color_p})

@app.callback(
    Output('order-status', 'figure'),
    Input('order-status', 'id')
)
def update_order_status(_):
    df = spark.sql("SELECT order_status, COUNT(*) AS total FROM order GROUP BY order_status").toPandas()
    fig = px.bar(df, x='order_status', y='total', color='order_status')
    fig.update_layout(
        plot_bgcolor=color_s,       # Fond du graphique
        paper_bgcolor=color_s,  # Fond extérieur (peut être blanc, gris, etc.)
        font=dict(color='white')            # Couleur du texte
    )
    return fig

@app.callback(
    Output('delivery-delay', 'figure'),
    Input('delivery-delay', 'id')
)
def update_delay(_):
    df_pd = df_delivery.select("late_delivery_risk").toPandas()
    counts = df_pd["late_delivery_risk"].value_counts().sort_index()
    labels = ["On Time", "Late"]
    fig = px.pie(values=counts, names=labels)
            # 🎨 Personnalisation du fond
    fig.update_layout(
        plot_bgcolor=color_s,       # Fond du graphique
        paper_bgcolor=color_s,  # Fond extérieur (peut être blanc, gris, etc.)
        font=dict(color='white')            # Couleur du texte
    )

    return fig

@app.callback(
    Output('geo-map', 'figure'),
    Input('geo-map', 'id')
)
def update_map(_):
    df_pd = df_customer.select("latitude", "longitude", "customer_segment").toPandas()

    # Nettoyage des coordonnées géographiques
    df_pd['latitude'] = pd.to_numeric(df_pd['latitude'], errors='coerce')
    df_pd['longitude'] = pd.to_numeric(df_pd['longitude'], errors='coerce')
    fig = px.scatter_map(df_pd, lat="latitude", lon="longitude", color="customer_segment",
                            zoom=1, height=500)
                # 🎨 Personnalisation du fond
    fig.update_layout(
        plot_bgcolor=color_s,       # Fond du graphique
        paper_bgcolor=color_s,  # Fond extérieur (peut être blanc, gris, etc.)
        font=dict(color='white')            # Couleur du texte
    )
    return fig

@app.callback(
    Output('country-orders', 'figure'),
    Input('country-orders', 'id')
)
def update_country_orders(_):
    df_geo = df_order.select("order_country").toPandas().dropna()
    df_geo["iso3"] = df_geo["order_country"].apply(lambda x: pycountry.countries.get(name=x).alpha_3 if pycountry.countries.get(name=x) else None)
    df_geo = df_geo.dropna(subset=["iso3"])
    orders_by_country = df_geo.groupby("iso3").size().reset_index(name="order_count")
    fig = px.choropleth(orders_by_country, locations="iso3", color="order_count",
                        color_continuous_scale="Blues", height=500)
                # 🎨 Personnalisation du fond
    fig.update_layout(
        plot_bgcolor=color_s,       # Fond du graphique
        paper_bgcolor=color_s,  # Fond extérieur (peut être blanc, gris, etc.)
        font=dict(color='white')            # Couleur du texte
    )
    return fig

@app.callback(
    Output('sales-time', 'figure'),
    Input('sales-time', 'id')
)
def update_sales_time(_):
    df_plot = spark.sql("""
    SELECT 
        o.`order_date_(dateorders)`,
        t.`sales_per_customer`,
        t.`benefit_per_order`
    FROM 
        order o
    JOIN 
        transaction t
    ON 
        o.order_id = t.order_id
""")

    df_order_pd = df_plot.toPandas()
    # ✅ 1. Conversion de la date si ce n’est pas déjà fait
    df_order_pd['order_date_(dateorders)'] = pd.to_datetime(df_order_pd['order_date_(dateorders)'])

    # ✅ 2. Grouper les données par mois
    sales_time = df_order_pd.groupby(pd.Grouper(key='order_date_(dateorders)', freq='ME')).agg({
        'sales_per_customer': 'sum',
        'benefit_per_order': 'sum'
    }).reset_index()

    fig = px.line(
        sales_time,
        x='order_date_(dateorders)',
        y=['sales_per_customer', 'benefit_per_order'],
        labels={'value': 'Montant (€)', 'order_date_(dateorders)': 'Date'},
        markers=True
    )
    fig.update_layout(legend_title_text='Indicateur')
                # 🎨 Personnalisation du fond
    fig.update_layout(
        plot_bgcolor=color_s,       # Fond du graphique
        paper_bgcolor=color_s,  # Fond extérieur (peut être blanc, gris, etc.)
        font=dict(color='white')            # Couleur du texte
    )
    return fig

@app.callback(
    Output('top-products', 'figure'),
    Input('top-products', 'id')
)
def update_top_products(_):
    top_products = df_product.join(df_orderitem, on="product_card_id") \
        .groupBy("product_name") \
        .sum("order_item_quantity") \
        .withColumnRenamed("sum(order_item_quantity)", "Total Quantity") \
        .orderBy("Total Quantity", ascending=False) \
        .limit(10)

    top_products_pd = top_products.toPandas()

    fig = px.bar(
        top_products_pd.sort_values("Total Quantity"),
        x="Total Quantity", y="product_name",
        orientation="h"
    )
    # 🎨 Personnalisation du fond
    fig.update_layout(
        plot_bgcolor=color_s,       # Fond du graphique
        paper_bgcolor=color_s,  # Fond extérieur (peut être blanc, gris, etc.)
        font=dict(color='white')            # Couleur du texte
    )
    return fig

@app.callback(
    Output('pic-products-heure', 'figure'),
    Input('pic-products-heure', 'id')
)
def update_top_products(_):
    hourly_counts = df_logs.groupBy("Hour").count().orderBy("Hour")
    # Pour visualiser avec matplotlib si besoin
    hourly_pd = hourly_counts.toPandas()

    fig = px.bar(
        hourly_pd,
        x="Hour", y="count",
        orientation="v"
    )
    # 🎨 Personnalisation du fond
    fig.update_layout(
        plot_bgcolor=color_s,       # Fond du graphique
        paper_bgcolor=color_s,  # Fond extérieur (peut être blanc, gris, etc.)
        font=dict(color='white')            # Couleur du texte
    )
    return fig

@app.callback(
    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    Input('interval-component', 'n_intervals')
)
def update_table(n):
    try:
        # Lire depuis HDFS (Parquet)
        df = spark.read.parquet("hdfs://localhost:9000/projet/predictions")
        
        if df.count() == 0:
            return [], []
        
        # Limiter pour éviter des surcharges
        pandas_df = df.toPandas()
        pandas_df = pandas_df[["type","order_id","product_name","benefit_per_order","sales_per_customer","days_for_shipping_(real)","days_for_shipment_(scheduled)","late_delivery_risk"]]
        # Générer les colonnes dynamiquement
        columns = [{"name": col, "id": col} for col in pandas_df.columns]
        data = pandas_df.to_dict('records')
        return data, columns

    except Exception as e:
        print("Erreur lors de la lecture des données HDFS:", str(e))
        return [], []

@app.callback(
    Output('delivery-delay_real', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_delay(_):
    # Chargement des données depuis HDFS
    df = spark.read.parquet("hdfs://localhost:9000/projet/predictions")

    # Si le DataFrame est vide, ne rien faire
    if df.count() == 0:
        fig = px.pie(values=[0])
        return fig

    # Conversion en Pandas pour analyse
    df_pd = df.select("late_delivery_risk").toPandas()

    # Comptage des valeurs
    counts = df_pd["late_delivery_risk"].value_counts().sort_index()
    
    # Pour éviter une erreur si certaines valeurs (0 ou 1) sont absentes
    values = [counts.get(0, 0), counts.get(1, 0)]
    labels = ["On Time", "Late"]

    # Création du graphique
    fig = px.pie(values=values, names=labels)

    # Personnalisation
    fig.update_layout(
        plot_bgcolor=color_s,
        paper_bgcolor=color_s,
        font=dict(color='white')
    )

    return fig


@app.callback(
    Output('data-table_segmentations', 'data'),
    Output('data-table_segmentations', 'columns'),
    Input('interval-component', 'n_intervals')
)
def update_table(n):
    try:
        # Lire depuis HDFS
        df = spark.read.parquet("hdfs://localhost:9000/projet/predictions_segment")

        if df.count() == 0:
            return [], []
        # Limiter le volume

        pandas_df = df.toPandas()
        pandas_df = pandas_df.drop_duplicates(["order_item_id"])
        pandas_df = pandas_df[["customer_id", "order_id", "customer_city","order_date_(dateorders)",
        "product_name","order_item_quantity","product_price","shipping_mode","cluster"]]
        

        # Générer les colonnes dynamiquement
        columns = [{"name": col, "id": col} for col in pandas_df.columns]
        data = pandas_df.to_dict('records')

        return data, columns

    except Exception as e:
        print("Erreur lors de la lecture des données HDFS:", str(e))
        return [], []

@app.callback(
    Output('segmentations_pie', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_delay(_):
     # Chargement des données depuis HDFS
    df = spark.read.parquet("hdfs://localhost:9000/projet/predictions_segment")

    # Si le DataFrame est vide ou la colonne cluster absente

    if df.count() == 0:
        fig = px.pie(values=[0])
        return fig
    
    cluster = df.groupBy("cluster").count().orderBy(desc("count"))
    # Pour visualiser avec matplotlib si besoin
    cluster = cluster.toPandas()

    fig = px.bar(
        cluster,
        x="cluster",y="count", 
        orientation="v"
    )
    # 🎨 Personnalisation du fond
    fig.update_layout(
        plot_bgcolor=color_s,       # Fond du graphique
        paper_bgcolor=color_s,  # Fond extérieur (peut être blanc, gris, etc.)
        font=dict(color='white')            # Couleur du texte
    )
    return fig

@app.callback(
    Output('data-table-logs', 'data'),
    Output('data-table-logs', 'columns'),
    Input('interval-component', 'n_intervals')
)
def update_table(n):
    try:
        # Lire depuis HDFS (Parquet)
        df = spark.read.parquet("hdfs://localhost:9000/projet/streamed_logs")
        
        if df.count() == 0:
            return [], []
        
        # Limiter pour éviter des surcharges
        pandas_df = df.toPandas()
        # Générer les colonnes dynamiquement
        columns = [{"name": col, "id": col} for col in pandas_df.columns]
        data = pandas_df.to_dict('records')
        return data, columns

    except Exception as e:
        print("Erreur lors de la lecture des données HDFS:", str(e))
        return [], []
    
@app.callback(
    Output('top-product-visite', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_delay(_):
    # Chargement des données depuis HDFS
    df = spark.read.parquet("hdfs://localhost:9000/projet/streamed_logs")

    if df.count() == 0:
        fig = px.pie(values=[0])
        return fig
    
    product_views = df.groupBy("Product").count().orderBy(desc("count"))
    # Pour visualiser avec matplotlib si besoin
    product_views = product_views.toPandas()

    fig = px.bar(
        product_views,
        x="count", y="Product",
        orientation="h"
    )
    # 🎨 Personnalisation du fond
    fig.update_layout(
        plot_bgcolor=color_s,       # Fond du graphique
        paper_bgcolor=color_s,  # Fond extérieur (peut être blanc, gris, etc.)
        font=dict(color='white')            # Couleur du texte
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
