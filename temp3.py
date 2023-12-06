from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to one-hot encoded format
y_train_one_hot = to_categorical(y_train, num_classes=10)
y_test_one_hot = to_categorical(y_test, num_classes=10)

# Define the ANN model architecture
model = Sequential([
    Flatten(input_shape=(28, 28)),
    #Dense(128, activation='relu'),  # First hidden layer with ReLU activation
    #Dense(64, activation='relu'),   # Second hidden layer with ReLU activation
    Dense(10, activation='sigmoid')  # Output layer with linear activation
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train_one_hot, epochs=3, validation_data=(x_test, y_test_one_hot))

# Summary of the model
model.summary()

test_loss, test_accuracy = model.evaluate(x_test, y_test_one_hot, verbose=0)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

model.save('temp3.h5')

