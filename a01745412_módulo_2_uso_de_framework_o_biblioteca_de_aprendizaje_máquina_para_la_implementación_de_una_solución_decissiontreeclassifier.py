#DecissionTreeClassifier - A01745412

#Importación de librerías y carga del dataset

#Se importan las bibliotecas necesarias para el procesamiento de datos, la creación del modelo de árbol de decisión, y las métricas de evaluación. Además, se carga el dataset desde el archivo CSV.


# Importación de librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Titanic.csv')
df.head()

#Limpieza y preparación de los datos
#Se eliminan las columnas irrelevantes, se manejan los valores nulos y se convierten las variables categóricas en numéricas. Esto es necesario para que el algoritmo de aprendizaje automático pueda procesar los datos correctamente.


# Verificación de valores faltantes antes de la limpieza
print("Valores nulos antes de la limpieza:\n", df.isnull().sum())

# Eliminación de columnas irrelevantes y llenamar valoresnulos
df_clean = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df_clean['Age'].fillna(df_clean['Age'].mean(), inplace=True)
df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)

# Verificar los valores faltantes después de la limpieza
print("Valores nulos después de la limpieza:\n", df_clean.isnull().sum())

# Convertir las variables categóricas ('Sex', 'Embarked') en numéricas
label_encoder = LabelEncoder()
df_clean['Sex'] = label_encoder.fit_transform(df_clean['Sex'])
df_clean['Embarked'] = label_encoder.fit_transform(df_clean['Embarked'])

#División de los datos en Train, Validation y Test
#El conjunto de datos se divide en tres partes: un 70% para entrenamiento, un 15% para validación, y un 15% para prueba. Esto ayuda a entrenar el modelo y luego evaluarlo en datos no utilizados durante el entrenamiento.

# Separar las características (X) de la variable objetivo (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Dividir el dataset en conjuntos de entrenamiento (70%), validación (15%) y prueba (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# División del dataset en características (X) y variable objetivo (y)
X = df_clean.drop('Survived', axis=1)
y = df_clean['Survived']

# División del dataset en entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Verificación delas dimensiones de los conjuntos de datos
print(f'Dimensiones de X_train: {X_train.shape}')
print(f'Dimensiones de X_val: {X_val.shape}')
print(f'Dimensiones de X_test: {X_test.shape}')

# Visualización de la distribución de la variable objetivo en cada subconjunto
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.countplot(x=y_train)
plt.title('Distribución en el conjunto de Entrenamiento')

plt.subplot(1, 3, 2)
sns.countplot(x=y_val)
plt.title('Distribución en el conjunto de Validación')

plt.subplot(1, 3, 3)
sns.countplot(x=y_test)
plt.title('Distribución en el conjunto de Prueba')

plt.tight_layout()
plt.show()

#Entrenamiento del modelo de árbol de decisión
#Se crea un modelo de árbol de decisión con una profundidad máxima para evitar overfitting. Luego se entrena utilizando el conjunto de entrenamiento.
# Entrenamiento del modelo de árbol de decisión con una profundidad máxima para evitar overfitting
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

#Realización de predicciones en el conjunto de validación
#Se realizan predicciones en el conjunto de validación que no fue utilizado para el entrenamiento.


# Predicciones en el conjunto de validación
y_val_pred = clf.predict(X_val)

#Evaluación del modelo

#Se evalúa el rendimiento del modelo utilizando varias métricas como la exactitud, precisión, recall, y el puntaje F1. También se genera y visualiza la matriz de confusión.


# Evaluación del modelo usando varias métricas
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

# Matriz de confusión
conf_matrix = confusion_matrix(y_val, y_val_pred)

# Métricas de evaluación
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

#Visualización de la matriz de confusión
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de confusión - Validación')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

#Visualización del árbol de decisión
#Haciendo uso de la función plot_tree de sklearn se presenta el Árbol de Decisión ya entrenado.


# Visualización del árbol de decisión entrenado
from sklearn import tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True, fontsize=10)
plt.title('Árbol de Decisión - Titanic')
plt.show()

# Generación de curvas de aprendizaje para analizar la varianza
train_scores = []
val_scores = []

# Probar diferentes profundidades del árbol
for depth in range(1, 20):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    # Precisión en entrenamiento y validación
    train_scores.append(model.score(X_train, y_train))
    val_scores.append(model.score(X_val, y_val))

# Visualización de las curvas de aprendizaje
plt.figure(figsize=(10, 6))
plt.plot(range(1, 20), train_scores, label='Precisión en Entrenamiento')
plt.plot(range(1, 20), val_scores, label='Precisión en Validación')
plt.xlabel('Profundidad del Árbol')
plt.ylabel('Precisión')
plt.title('Curvas de Aprendizaje')
plt.legend()
plt.show()

# Ajuste del modelo con regularización
model_regularizado = DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
model_regularizado.fit(X_train, y_train)

# Predicciones y evaluación después de la regularización
y_val_pred_reg = model_regularizado.predict(X_val)

# Matriz de confusión después de la regularización
conf_matrix_reg = confusion_matrix(y_val, y_val_pred_reg)

# Visualización de las métricas de evaluación después de la regularización
accuracy_reg = accuracy_score(y_val, y_val_pred_reg)
precision_reg = precision_score(y_val, y_val_pred_reg)
recall_reg = recall_score(y_val, y_val_pred_reg)
f1_reg = f1_score(y_val, y_val_pred_reg)

print(f'Accuracy (Regularización): {accuracy_reg:.4f}')
print(f'Precision (Regularización): {precision_reg:.4f}')
print(f'Recall (Regularización): {recall_reg:.4f}')
print(f'F1 Score (Regularización): {f1_reg:.4f}')

# Visualización de las matrices de confusión antes y después de la regularización
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Antes de la Regularización')
ax[0].set_xlabel('Predicción')
ax[0].set_ylabel('Real')

sns.heatmap(conf_matrix_reg, annot=True, fmt='d', cmap='Blues', ax=ax[1])
ax[1].set_title('Después de la Regularización')
ax[1].set_xlabel('Predicción')
ax[1].set_ylabel('Real')

plt.show()

# Visualización del árbol de decisión entrenado con regularización
plt.figure(figsize=(20, 10))
tree.plot_tree(model_regularizado, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True, fontsize=10)
plt.title('Árbol de Decisión - Titanic (Después de la Regularización)')
plt.show()