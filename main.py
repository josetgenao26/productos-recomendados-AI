import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tabulate import tabulate
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import joblib

# Carga tus datos desde un archivo CSV
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/transacciones-productos-transform.csv")

# Elimina filas con valores faltantes (NaN)
data = data.dropna()

# Asegúrate de que haya suficientes filas después de eliminar valores faltantes
if len(data) > 0:
    # Comprueba si ya tienes un modelo K-Means entrenado previamente
    try:
        kmeans = joblib.load("kmeans_model.pkl")
    except FileNotFoundError:
        # Si el modelo no existe, entrena uno y guárdalo
        num_clusters = 3
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(data.drop(['Transacción'], axis=1))
        joblib.dump(kmeans, "kmeans_model.pkl")  # Guarda el modelo en un archivo

    # Permitir al usuario ingresar un producto para obtener recomendaciones
    producto_ingresado = input("Ingresa el nombre del producto para obtener recomendaciones: ")

    if producto_ingresado in data.columns:
        # Encuentra el cluster al que pertenece el producto ingresado
        cluster_producto = kmeans.predict(data[data[producto_ingresado] == 1].drop(['Transacción'], axis=1))[0]

        # Encuentra los productos más populares en el mismo cluster
        cluster_product_counts = data.drop(['Transacción'], axis=1).groupby(kmeans.labels_).sum()
        productos_populares = cluster_product_counts.loc[cluster_producto]

        # Permite al usuario definir cuántos productos desea como recomendación
        cantidad_recomendaciones = int(input("Ingresa la cantidad de productos recomendados que deseas: "))

        # Filtra los productos recomendados
        productos_recomendados = productos_populares[productos_populares > 0].index.tolist()[:cantidad_recomendaciones]

        print(f"Recomendaciones de productos relacionados para {producto_ingresado}:")
        print(productos_recomendados)
    else:
        print("El producto ingresado no se encuentra en los datos.")
else:
    print("No hay suficientes datos después de eliminar valores faltantes.")
