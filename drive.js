const parseArgs = require('minimist');
const io = require('socket.io')();
const tf = require('@tensorflow/tfjs-node');

const { model: modelDir = 'model', speed: maxSpeed = 30 } = parseArgs(
  process.argv.slice(2)
);

tf.loadLayersModel(`file://${modelDir}/model.json`).then((model) => {
  io.on('connection', (socket) => {
    console.log('Simulator connected');

    socket.on('telemetry', (telemetry) => {
      if (!telemetry) return;

      const imageBuffer = Buffer.from(telemetry.image, 'base64');
      const imageTensor = tf.node
        .decodeJpeg(imageBuffer)
        .div(255)
        .reshape([1, 160, 320, 3]);
      const steering = model.predict(imageTensor).squeeze().arraySync();

      const throttle = 1 - telemetry.speed / maxSpeed;

      console.log('steering_angle:', steering);

      socket.emit('steer', {
        steering_angle: String(steering),
        throttle: String(throttle)
      });
    });
  });

  io.listen(4567);
});
