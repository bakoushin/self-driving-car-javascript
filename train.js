const fs = require('fs');
const path = require('path');
const parseArgs = require('minimist');
const linesCount = require('file-lines-count');
const csv = require('csv-parser');
const tf = require('@tensorflow/tfjs-node');

const {
  data: dataDir = 'data',
  model: modelDir = 'model',
  epochs = 10
} = parseArgs(process.argv.slice(2));

const pathToCSV = path.join(dataDir, 'driving_log.csv');

async function* dataGenerator() {
  while (true) {
    const csvStream = fs
      .createReadStream(pathToCSV)
      .pipe(
        csv([
          'center',
          'left',
          'right',
          'steering',
          'throttle',
          'brake',
          'speed'
        ])
      );

    for await (const { center, left, right, steering } of csvStream) {
      const centerImageBuffer = fs.promises.readFile(center);
      const leftImageBuffer = fs.promises.readFile(left);
      const rightImageBuffer = fs.promises.readFile(right);

      const offset = 0.333;

      yield [await centerImageBuffer, Number(steering)];
      yield [await leftImageBuffer, Number(steering) + offset];
      yield [await rightImageBuffer, Number(steering) - offset];
    }

    csvStream.destroy();
  }
}

async function initModel() {
  let model;

  try {
    model = await tf.loadLayersModel(`file://${modelDir}/model.json`);
    console.log(`Model loaded from: ${modelDir}`);
  } catch {
    model = tf.sequential({
      layers: [
        tf.layers.cropping2D({
          cropping: [
            [75, 25],
            [0, 0]
          ],
          inputShape: [160, 320, 3]
        }),
        tf.layers.conv2d({
          filters: 16,
          kernelSize: [3, 3],
          strides: [2, 2],
          activation: 'relu'
        }),
        tf.layers.maxPool2d({ poolSize: [2, 2] }),
        tf.layers.conv2d({
          filters: 32,
          kernelSize: [3, 3],
          strides: [2, 2],
          activation: 'relu'
        }),
        tf.layers.maxPool2d({ poolSize: [2, 2] }),
        tf.layers.flatten(),
        tf.layers.dense({ units: 1024, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.25 }),
        tf.layers.dense({ units: 128, activation: 'relu' }),
        tf.layers.dense({ units: 1, activation: 'linear' })
      ]
    });
  }

  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  return model;
}

(async function () {
  const batchSize = 64;

  const dataset = tf.data
    .generator(dataGenerator)
    .map(([imageBuffer, steering]) => {
      const xs = tf.node.decodeJpeg(imageBuffer).div(255);
      const ys = tf.tensor1d([steering]);
      return { xs, ys };
    })
    .shuffle(batchSize)
    .batch(batchSize);

  const model = await initModel();

  const totalSamples = (await linesCount(pathToCSV)) * 3;

  await model.fitDataset(dataset, {
    epochs,
    batchesPerEpoch: Math.floor(totalSamples / batchSize)
  });

  await model.save(`file://${modelDir}`);

  console.log(`Model saved to: ${modelDir}`);
})();
