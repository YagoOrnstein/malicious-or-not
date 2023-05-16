# Bu kod, bir veri setini yükleyip işleyerek bir sinir ağı modelini eğitir ve ardından sınıflandırma performansını değerlendirir.
# Bu kod bloğunda, kullanılacak olan Python kütüphaneleri içe aktarılır. Bu kütüphaneler, veri işleme, görselleştirme, makine öğrenimi ve derin öğrenme işlemleri için kullanılacak olan fonksiyonları sağlar.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import load_model, Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Bu kod, "maliciousornot.xlsx" adlı Excel dosyasından veri setini yükler ve bir DataFrame nesnesine atar.
df = pd.read_excel("maliciousornot.xlsx")


df.corr()["Type"].sort_values()
sns.countplot(data=df, x="Type")
plt.show()
# Bu kod bloğu, "Type" özelliği ile diğer özellikler arasındaki korelasyonu hesaplar ve sıralar. Ardından, veri setindeki "Type" özelliğinin sınıf dağılımını çubuk grafiğiyle görselleştirir.


X = df.drop("Type", axis=1).values
y = df["Type"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Bu kod bloğunda, veri seti özellikleri (X) ve hedef değişken (y) ayrıştırılır. Ardından, veri seti, eğitim ve test kümelerine bölünür. Min-Max ölçeklendirme yöntemi kullanılarak özellik değerleri 0 ile 1 arasında ölçeklenir.


X_train.shape  # (383, 30)

model = Sequential()
model.add(Dense(units=30, activation="relu"))
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")

model.fit(x=X_train, y=y_train, epochs=700,
          validation_data=(X_test, y_test), verbose=1)
# Bu kod bloğunda, keras kütüphanesini kullanarak bir sinir ağı modeli oluşturulur. Modelin katmanları ve aktivasyon fonksiyonları tanımlanır. Ardından, model derlenir ve eğitim verileri kullanılarak belirli bir sayıda epoch üzerinde eğitilir. Eğitim sırasında doğrulama verileri de kullanılır.


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.show()
# Bu kod bloğu, eğitim sürecinde elde edilen kayıp (loss) değerlerini içeren bir DataFrame oluşturur ve bunu çizgi grafiği olarak görselleştirir.

model = Sequential()
model.add(Dense(units=30, activation="relu"))
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=15, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")

early_stop = EarlyStopping(
    monitor="val_loss", patience=25, mode="min", verbose=1)

model.fit(X_train, y_train, epochs=700, validation_data=(
    X_test, y_test), verbose=1, callbacks=[early_stop])
# Bu kod bloğunda, aynı sinir ağı modeli bir kez daha oluşturulur ve eğitilir. Ayrıca, EarlyStopping geri çağrısı kullanılarak eğitim süreci erken durdurulabilir.


model_loss2 = pd.DataFrame(model.history.history)
model_loss2.plot()
plt.show()
# Bu kod bloğu, ikinci eğitim sürecinde elde edilen kayıp (loss) değerlerini içeren bir DataFrame oluşturur ve bunu çizgi grafiği olarak görselleştirir.

model = Sequential()
model.add(Dense(units=30, activation="relu"))
model.add(Dropout(rate=0.5))
model.add(Dense(units=15, activation="relu"))
model.add(Dropout(rate=0.5))
model.add(Dense(units=15, activation="relu"))
model.add(Dropout(rate=0.5))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam")

model.fit(X_train, y_train, epochs=700, validation_data=(
    X_test, y_test), verbose=1, callbacks=[early_stop])
# Bu kod bloğunda, dropout katmanları eklenerek regularizasyon uygulanan bir sinir ağı modeli oluşturulur ve eğitilir.

model_loss3 = pd.DataFrame(model.history.history)
model_loss3.plot()
plt.show()
# Bu kod bloğu, üçüncü eğitim sürecinde elde edilen kayıp (loss) değerlerini içeren bir DataFrame oluşturur ve bunu çizgi grafiği olarak görselleştirir.

predictions = model.predict(X_test)
classes = np.argmax(predictions, axis=1)

print(classification_report(y_test, classes))
confusion_matrix(y_test, classes)
# Bu kod bloğunda, eğitilmiş model kullanılarak test verileri üzerinde tahminler yapılır. Ardından, sınıflandırma performansını değerlendirmek için sınıflandırma raporu (classification_report) ve karışıklık matrisi (confusion_matrix) hesaplanır ve ekrana yazdırılır.

# Bu kod, bir veri setinin yüklenmesi, ön işleme adımlarının uygulanması ve ardından sinir ağı modellerinin oluşturulması ve eğitimi ile performans değerlendirmesini içeren bir veri bilimi iş akışını temsil etmektedir. Her adımda yapılan işlemler açıklanmış ve sonuçlar görselleştirilmiştir.
