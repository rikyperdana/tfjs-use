// Uncomment these if it is run through node
// import * as tf from "@tensorflow/tfjs"
// import * as use from "@tensorflow-models/universal-sentence-encoder"

const clog = console.log
const withAs = (obj, cb) => cb(obj)
const allDone = (arr, cb) => Promise.all(arr).then(cb)
const getHalf = (side, arr) => ({
  left: arr.slice(0, arr.length / 2),
  right: arr.slice(arr.length / 2, arr.length)
})[side]

const csvFile = "https://raw.githubusercontent.com/rikyperdana/ddilia/main/ddi_tiny.csv"

const getData =
  (csv, label, cb) => tf.data.csv(
    csv, {columnConfigs: {
      [label]: {isLabel: true}
    }}
  ).map(({xs, ys}) => ({
    xs: Object.values(xs).join(' '),
    ys: Object.values(ys)[0]
  })).toArray().then(cb)

const embedAll = (csv, label, cb) =>
  use.load().then(embedder => getData(
    csv, label, rows => allDone([
      ...rows.map(({xs}) => embedder.embed(xs)),
      ...rows.map(({ys}) => embedder.embed(ys))
    ]).then(result => cb(tf.data.zip({
      xs: tf.data.array(getHalf('left', result)),
      ys: tf.data.array(getHalf('right', result))
    }).batch(5)))
  ))

const brain = tf.sequential({layers: [
  tf.layers.dense({
    activation: 'relu', units: 128,
    inputShape: [512]
  }),
  tf.layers.dense({
    activation: 'linear', units: 512
  })
]})

brain.compile({
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
})

const trainData =
  (csv, label, model, cb) => embedAll(
    csv, label, dataset => model.fitDataset(
      dataset, {epochs: 1}
    ).then(cb)
  )

const textData = "in a study of 11 hiv-infected patients receiving drug1-maintenance therapy ( 40 mg and 90 mg daily ) with 600 mg of drug2 twice daily ( twice the currently recommended dose )  oral drug0 clearance increased 22 % ( 90 % ci 6 % to 42 % )"

0 && use.load().then(
  embedder => embedder.embed(textData).then(
    embededText => trainData(
      csvFile, 'ddi_type', brain,
      done => clog(
        brain.predict(embededText)
        .arraySync()[0]
      )
    )
  )
)