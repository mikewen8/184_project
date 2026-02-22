#https://www.tensorflow.org/guide/keras/sequential_model
#--------Explanation on what hte Sequential model does
# two ways to define the model
"""
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)

model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))

# remove a layers
model.pop()
print(len(model.layers))  # 2

# simple terms
https://www.geeksforgeeks.org/deep-learning/difference-between-ann-cnn-and-rnn/
explaines the covolutions
https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
"""