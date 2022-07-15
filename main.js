const { tensor } = require('./libs/main');

console.log( tensor );

A = new Math.tensor(5,5);
A.order();
A.show();

B = A.clone()
B.shape(2,10);
B.show(' ');

B.sin().show(' ',2);