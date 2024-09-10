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

const getData = (csv, cb) => tf.data.csv(csv).map(
  ({text, drug1, drug2, ddi, ddi_type}) => ({
    inputs: [text, drug1, drug2, ddi].join(''),
    output: ddi_type
  })
).toArray().then(cb)

const embedAll = (csv, cb) => use.load().then(
  embedders => getData(csv, dataset => allDone([
    ...dataset.map(i => embedders.embed(i.inputs)),
    ...dataset.map(i => embedders.embed(i.output))
  ], embededs => cb(tf.data.zip({
    xs: tf.data.array(getHalf('left', embededs)),
    ys: tf.data.array(getHalf('right', embededs))
  }))))
)

const trainModel =
  (csv, model, cb) => embedAll(
    csv, dataset => model.fitDataset(
      dataset, {epochs: 1}
    ).then(done => cb(model))
  )

const textData = "in a study of 11 hiv-infected patients receiving drug1-maintenance therapy ( 40 mg and 90 mg daily ) with 600 mg of drug2 twice daily ( twice the currently recommended dose )  oral drug0 clearance increased 22 % ( 90 % ci 6 % to 42 % )"

