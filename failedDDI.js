// Source of dataset in csv form
const ddiTiny = 'https://raw.githubusercontent.com/rikyperdana/ddilia/main/ddi_tiny.csv'

// Helper functions
const getHalf = (side, arr) => ({
  left: arr.slice(0, arr.length / 2),
  right: arr.slice(arr.length / 2, arr.length)
})[side]

// Initiate model
const brain = tf.sequential({layers: [
  tf.layers.dense({
    activation: 'relu', units: 32,
    inputShape: [1, 512]
  }),
  tf.layers.dense({
    activation: 'softmax', units: 10
  })
]})

brain.compile({
  optimizer: 'sgd',
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
})

// From csv to tensors ------------------------
use.load().then( // load the embedding model
  model => tf.data.csv(ddiTiny).map(i => ({
    // set the inputs and output
    inputs: [i.text, i.drug1, i.drug2].join(','),
    output: i.ddi_type
  })).toArray().then(temp => Promise.all([
    // convert both of them into tensors
    ...temp.map(i => model.embed(i.inputs)),
    ...temp.map(i => model.embed(i.output))
    // when done, store into global var
  ]).then(res => brain.fitDataset(
    // what is supposed to be here
    tf.data.zip({
      xs: tf.data.array(getHalf('left', res)),
      ys: tf.data.array(getHalf('right', res)),
    }),
    {epochs: 1}
  ).then(console.log)))
)





// saveCSV('kk', ['coba,satu,dulu\nyang,ini,juga'])
const saveCSV = (title, contents) => saveAs(
  new Blob(contents, {type: 'text/csv;charset=utf-8'}),
  `${title}.csv`
)