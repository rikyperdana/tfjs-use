import * as tf from "@tensorflow/tfjs"
import * as use from "@tensorflow-models/universal-sentence-encoder"

const clog = console.log
const withAs = (obj, cb) => cb(obj)
const allDone = (arr, cb) => Promise.all(arr).then(cb)
const getHalf = (side, arr) => ({
  left: arr.slice(0, arr.length / 2),
  right: arr.slice(arr.length / 2, arr.lengt)
})[side]

const csvFile = "https://raw.githubusercontent.com/rikyperdana/ddilia/main/ddi_tiny.csv"
const sentences = [
  "I love you",
  "But, you don't love me"
]

const brain = tf.sequential({layers: [
  tf.layers.dense({
    units: 1, activation: 'relu',
    inputShape: [1, 512]
  }),
  tf.layers.dense({
    units: 1, activation: 'softmax'
  })
]})

brain.compile({
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
})

const getData = cb => tf.data.csv(csvFile).map(
  ({text, drug1, drug2, ddi, ddi_type}) => ({
    inputs: [text, drug1, drug2, ddi].join(''),
    output: ddi_type
  })
).toArray().then(cb)

use.load().then(
  embedders => getData(dataset => allDone([
    ...dataset.map(i => embedders.embed(i.inputs)),
    ...dataset.map(i => embedders.embed(i.output)),
  ], embededs => brain.fitDataset(
    tf.data.zip({
      xs: tf.data.array(getHalf('left', embededs)),
      ys: tf.data.array(getHalf('right', embededs))
    }), {epochs: 1}
  ).then(clog)))
)