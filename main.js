// BAGIAN DEPENDENSI --------------------------

// import * as tf from "@tensorflow/tfjs"
// import * as use from "@tensorflow-models/universal-sentence-encoder"

// BAGIAN FUNGSI -------------------------------

print = console.log

withAs = (obj, cb) => cb(obj)

allDone = (arr, cb) => Promise.all(arr).then(cb)

getHalf = (side, arr) => ({
  left: arr.slice(0, arr.length / 2),
  right: arr.slice(arr.length / 2, arr.length)
})[side]

getData =
  (csv, label, cb) => tf.data.csv(
    csv, {columnConfigs: {
      [label]: {isLabel: true}
    }}
  ).map(({xs, ys}) => ({
    xs: Object.values(xs).join(' '),
    ys: Object.values(ys)[0]
  })).toArray().then(cb)

embedOne = (text, cb) =>
  use.load().then(
    embedder => embedder
    .embed(text).then(cb)
  )

embedAll = (dataset, cb) =>
  use.load().then(embedder => allDone([
    ...dataset.map(({xs}) => embedder.embed(xs)),
    ...dataset.map(({ys}) => embedder.embed(ys))
  ]).then(result => cb(tf.data.zip({
    xs: tf.data.array(getHalf('left', result)),
    ys: tf.data.array(getHalf('right', result))
  }).batch(1))))

brain = tf.sequential({layers: [
  tf.layers.dense({
    activation: 'relu', units: 512,
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

trainModel = (embededs, model, cb) =>
  model.fitDataset(embededs, {epochs: 1})
  .then(done => cb(done, model))

saveModel = (model, name, cb) =>
  model.save(`downloads://${name}`).then(cb)

similarityTest = (yPred, yTrue) =>
  tf.losses.huberLoss(yPred, yTrue).dataSync()[0]

// BAGIAN TESTING --------------------------------

// Untuk melatih model dan simpan model
latih_dan_simpan = x => getData(
  // 1. Tarik dulu csv nya
  csvFile, 'ddi_type',
  dataset => embedAll(
    // 2. Embed semua teks jadi tensor
    dataset,
    embededs => trainModel(
      // 3. Latih model dengan embeddings tadi
      embededs, brain,
      (done, model) => saveModel(
        // 4. Simpan sebagai eksperimen
        model, 'eksperimen3', print
      )
    )
  )
)

// Untuk melatih model lalu prediksi
latih_dan_tes = x => getData(
  csvFile, 'ddi_type', dataset => embedAll(
    // 1. Ambil data set nya
    dataset, embededs => trainModel(
      // 2. Train model
      embededs, brain, done => embedOne(
        // 3. Embed test sentence
        testSentence, embededTest => print(
          // 4. Predict
          brain.predict(embededTest)
          .arraySync()
        )
      )
    )
  )
)