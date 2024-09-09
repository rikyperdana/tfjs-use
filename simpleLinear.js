const
xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]),
ys = tf.tensor2d([-3,-1, 1, 3, 5, 7], [6, 1]),
model = tf.sequential()

model.add(tf.layers.dense({units: 1, inputShape: [1]}))
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'})
model.fit(xs, ys, {epochs: 250}).then(
  x => console.log(model.predict(
    tf.tensor2d([20], [1, 1])
  ).dataSync()),
)