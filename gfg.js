import * as tf from "@tensorflow/tfjs"

const csvFile = "https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv"
const model = tf.sequential({layers: [
  tf.layers.dense({
    units: 1, inputShape: [12]
  })
]})

model.compile({
  optimizer: tf.train.sgd(0.0001),
  loss: 'meanSquaredError'
})

const dataset = tf.data.csv(
  csvFile, {columnConfigs: {
    tax: {isLabel: true}
  }}
).map(({xs, ys}) => ({
  // xs: Object.values(xs),
  xs: tf.tensor(Object.values(xs), [12]),
  ys: tf.tensor(Object.values(ys), [1])
})).batch(5)

const testData = [
  0.25915, 0, 21.89,
  0, 0.624, 5.693,
  96, 1.7883, 4,
  21.2, 17.19, 16.2
]

model.fitDataset(dataset, {epochs: 5})
.then(done => console.log(model.predict(
  tf.tensor(testData, [1, 12])
).dataSync()[0]))