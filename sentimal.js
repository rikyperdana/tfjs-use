// Import only necessary libraries
import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@4.20.0/+esm'
import * as backend from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu@4.20.0/+esm'
import * as USE from 'https://cdn.jsdelivr.net/npm/@tensorflow-models/universal-sentence-encoder@1.3.3/+esm'
import * as tr from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';

// Source of dataset in csv form
const ddiCsv = 'https://raw.githubusercontent.com/rikyperdana/ddilia/main/ddi_tiny.csv'

// Define the type of task to be piped
const pipeType = 'token-classification'

// Define specific models: optional
const SAmodel = 'Xenova/bert-base-multilingual-uncased-sentiment'
const nerModel = 'Xenova/bert-base-NER'

// The text to be examined
const text = "Hello my name is Riky and I'm a student"

// force load from huggingface
tr.env.allowLocalModels = false

// Transformers run the task
tr.pipeline(pipeType, nerModel)
.then(pipe => pipe(text))
.then(console.log)