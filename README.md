# Run a self-driving car using JavaScript and TensorFlow.js

This project demonstrates how to train a self-driving car to steer and to drive autonomously in [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) using [TensorFlow.js](https://www.tensorflow.org/js).

[<img src="https://img.youtube.com/vi/7fvKAR1TosA/maxresdefault.jpg" width="50%">](https://youtu.be/7fvKAR1TosA)

Featured on [TensorFlow](https://www.youtube.com/@TensorFlow) YouTube channel:

[<img src="https://img.youtube.com/vi/5C29fhFL_wQ/maxresdefault.jpg" width="50%">](https://youtu.be/5C29fhFL_wQ)

See also detalied explanation of this project in a Medium publication: [Run a self-driving car using JavaScript and TensorFlow.js](https://medium.com/p/run-a-self-driving-car-using-javascript-and-tensorflow-js-8b9b3f7af23d?sk=549ee7230aa2b05fead0e68210982cf8).

## Usage

1. Clone this project.
2. Download [Udacity Simulator](https://github.com/udacity/self-driving-car-sim) for Term 1.
3. Record data using the simulator.
4. Train the model using recorded data.
5. Run the simulator in autonomous mode.
6. Drive the car using the trained model.

## Cloning and initialization

```
git clone https://github.com/bakoushin/self-driving-car-javascript.git
cd self-driving-car-javascript
npm install
```

> Note: the code in this project is expected to run on Node.js 12 or greater.

## Training a model

```
node train.js [--data DATA_DIRECTORY] [--model [MODEL_DIRECTORY] [--epochs NUMBER_OF_EPOCHS]

# Example:

node train.js --data ~/Documents/track1 --epochs 3
```

Default values:

- `data` = 'data' directory within the project directory
- `model` = 'model' directory within the project directory
- `epochs` = 10

## Driving a car

```
node drive.js [--model MODEL_DIRECTORY] [--speed SPEED_LIMIT]

# Example:

node drive.js --speed 20

```

Default values:

- `model` = 'model' directory within the project directory
- `speed` = 30

## Author

Alex Bakoushin

## License

MIT
