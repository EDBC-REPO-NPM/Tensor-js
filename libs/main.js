
module.exports.tensor = require('./Math/tensor');

try{
	module.exports.kernel = require('./Math/kernel');
} catch(e) { }

/*
evalFile('./libs/Math/math.js');
evalFile('./libs/Math/image.js');
evalFile('./libs/Math/neuron.js');
*/
