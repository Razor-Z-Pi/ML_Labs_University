from tensorflow.keras.datasets import cifar10

# последовательная модель (стек слоев)
from tensorflow.keras.models import Sequential

# полносвязный слой и слой выпрямляющий матрицу в вектор
from tensorflow.keras.layers import Dense, Flatten

# слой выключения нейронов и слой нормализации выходных данных (нормализует данные в пределах текущей выборки)
from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D, GaussianDropout

# слои свертки и подвыборки
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

# работа с обратной связью от обучающейся нейронной сети
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# вспомогательные инструменты
from tensorflow.keras import utils

# работа с изображениями
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
%matplotlib inline

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
classes=['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка', 'лошадь', 'корабль', 'грузовик']



# Размер мини-выборки
batch_size = 128

# Количество классов изображений
nb_classes = 10

# Количество эпох для обучения
nb_epoch = 150

# Размер изображений
img_rows, img_cols = 512, 512

# Количество каналов в изображении: RGB
img_channels = 3

for n in range(1, 100):
  plt.imshow(X_train[n])
  plt.show()
  print("Номер класса:", y_train[n])
  print("Тип объекта:", classes[y_train[n][0]])



