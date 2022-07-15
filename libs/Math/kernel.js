
const GPU = require('gpu');
const _kernel = new Object(); 
Math.GPU = new GPU({ mode:'gpu' });


// TODO: Tensor Kernels ######################################################//
_kernel.flat = Math.GPU.createKernel( function(A){
	return A[this.thread.x];
}).setDynamicOutput(true);

_kernel.Tsp = Math.GPU.createKernel( function(A){
	return A[0][this.thread.x * A[1][0] + this.thread.y];
}).setDynamicOutput(true);

_kernel.Smul = Math.GPU.createKernel( function(A,B){
	return A[this.thread.x] * B;
}).setDynamicOutput(true);

_kernel.Sdiv = Math.GPU.createKernel( function(A,B){
	return A[this.thread.x] / B;
}).setDynamicOutput(true);

_kernel.Texp = Math.GPU.createKernel( function(A){
	return Math.exp( A[this.thread.x] );
}).setDynamicOutput(true);

_kernel.Tpow = Math.GPU.createKernel( function(A,B){
	return Math.pow( A[this.thread.x],B );
}).setDynamicOutput(true);

_kernel.Tadd = Math.GPU.createKernel( function(A,B){
	return A[this.thread.x] + B[this.thread.x];
}).setDynamicOutput(true);

_kernel.Tsub = Math.GPU.createKernel( function(A,B){
	return A[this.thread.x] - B[this.thread.x];
}).setDynamicOutput(true);

_kernel.Tmul = Math.GPU.createKernel( function(A,B){
	var sum = 0;
	for( var k=0; k<A[1][0]; k++ ){
		sum += A[0][ this.thread.y*A[1][0]+k ] * B[0][ k*B[1][0]+this.thread.x ];
	}	return sum;
}).setDynamicOutput(true);

_kernel.Tdot = Math.GPU.createKernel( function(A,B){
	return A[this.thread.x] * B[this.thread.x];
}).setDynamicOutput(true);


// TODO: Neuron Kernels ######################################################//
_kernel.cost = Math.GPU.createKernel( function(Y,A,kernel){
	var i = this.thread.x;
	switch(kernel){
	case  1: return Y[i]/A[i] - (1-Y[i])/(1-A[i]); break;
	default: return Y[i] - A[i]; break; }
}).setDynamicOutput(true);

_kernel.linear = Math.GPU.createKernel( function(X,dev){
	var sum = 0; if(dev == true) sum=X; 
	else sum=1; return sum;
}).setDynamicOutput(true);

_kernel.tangh = Math.GPU.createKernel( function(X,dev){
	var sum = 0;
	if(dev == true) sum=Math.tanh( X[this.thread.x] ); 
	else sum=1-( Math.pow(X[this.thread.x],2) ); 
	return sum;
}).setDynamicOutput(true);

_kernel.softPlus = Math.GPU.createKernel( function(X,dev){
	var sum = 0;
	if(dev == true) sum=Math.log(1+Math.exp(X[this.thread.x])); 
	else sum=1/(1+Math.exp(-X[this.thread.x])); 
	return sum;
}).setDynamicOutput(true);

_kernel.atang = Math.GPU.createKernel( function(X,dev){
	var sum = 0;
	if(dev == true) sum=Math.atan(X[this.thread.x]); 
	else sum=1/(1+Math.pow(X[this.thread.x],2)); 
	return sum;
}).setDynamicOutput(true);

_kernel.sigmoid = Math.GPU.createKernel( function(X,dev){
	var sum = 0;
	if(dev == true) sum=1/(1+Math.exp(-X[this.thread.x])); 
	else sum=X[this.thread.x]*(1-X[this.thread.x]); 
	return sum;
}).setDynamicOutput(true);

_kernel.binary = Math.GPU.createKernel( function(X,dev){
	var sum = 0;
	if(dev == true){
		if( X[this.thread.x] > 1 )
			sum = 1;
		else if( X[this.thread.x] < -1 )
			sum = -1
		else
			sum = X[this.thread.x]
	} else {
		if( -1 < X[this.thread.x] && X[this.thread.x] < 1 )
			sum = 1;
		else
			sum = 0;
	}
	return sum;
}).setDynamicOutput(true);

_kernel.relu = Math.GPU.createKernel( function(X,dev){
	var sum = 0;
	if(dev == true) 
		if(X[this.thread.x]>0) sum=X[this.thread.x];
		else sum=0;
	else 
		if(X[this.thread.x]>0) sum=1;
		else sum=0;		 
	return sum;
}).setDynamicOutput(true);

_kernel.softMax1 = Math.GPU.createKernel( function(X,sum){
	return Math.exp(X[this.thread.x])/sum;
}).setDynamicOutput(true);

_kernel.softMax2 = Math.GPU.createKernel( function(C,X){
	var j = this.thread.x;
	var i = this.thread.y;
	var acum = 0;

	if(i!=j) acum = C[i] + X[0][j];
	return acum;
}).setDynamicOutput(true);

_kernel.softMax3 = Math.GPU.createKernel( function(C,X){
	var acum = C[this.thread.x] * X[this.thread.x];
	return acum;
}).setDynamicOutput(true);

_kernel.gauss = Math.GPU.createKernel( function(X,Y,T,dev){
	var j = this.thread.x;
	var i = this.thread.y;
	var sum = 0;

	if(dev == true) sum = Math.exp(-Math.pow(X[j]-Y[i],2) / (2*Math.pow(T,2)));
	else sum = Math.exp(-Math.pow(X[j]-Y[i],2) / (2*Math.pow(T,2))) * ((X[j]-Y[i]) / (Math.pow(T,2))); 

	return sum;
}).setDynamicOutput(true);


// TODO: Image Kernels #######################################################//
_kernel.HaarEdgeVer = Math.GPU.createKernel( function(img,sx,sy){

	const i = this.thread.y;
	const j = this.thread.x;

	return(
	//filtro negro
	-	(
			img[0][ (i)* 	img[1][0] +(j) 	  ]    + //BA
			img[0][ (i)* 	img[1][0] +(j+sx) ]    + //BB
			img[0][ (i+sy)*	img[1][0] +(j)	  ]	   - //BC
			img[0][ (i+sy)* img[1][0] +(j+sx) ]		 //BD
		)	

	//filtro blanco
	+	(
			img[0][  (i+sy)*img[1][0] +(j)	  ]	   + //BC
			img[0][  (i+sy)*img[1][0] +(j+sx) ]	   + //BD
			img[0][(i+sy*2)*img[1][0] +(j)	  ]	   - //BE
			img[0][(i+sy*2)*img[1][0] +(j+sx) ]		 //BF
		)
	);

}).setDynamicOutput(true);

_kernel.HaarEdgeHor = Math.GPU.createKernel( function(img,sx,sy){

	var i = this.thread.y;
	var j = this.thread.x;
	
	return(
	//filtro negro
	-	(
			img[0][ (i)* 	img[1][0] +(j) 	  ]    + //BA
			img[0][ (i)* 	img[1][0] +(j+sx) ]    + //BB
			img[0][ (i+sy)*	img[1][0] +(j)	  ]	   - //BC
			img[0][ (i+sy)* img[1][0] +(j+sx) ]		 //BD
		)	

	//filtro blanco
	+	(
			img[0][ (i)*   img[1][0] +(j+sx)  ]    + //BB
			img[0][ (i)*   img[1][0] +(j+sx*2)]    + //BB
			img[0][ (i+sy)*img[1][0] +(j+sx)  ]	   - //BD
			img[0][ (i+sy)*img[1][0] +(j+sx*2)]	     //BE
		)		
	);

}).setDynamicOutput(true);

_kernel.HaarLineVer = Math.GPU.createKernel( function(img,sx,sy){

	var i = this.thread.y;
	var j = this.thread.x;

	return(
	//filtro negro
	-	(
			img[0][ (i)* 	img[1][0] +(j) 	  ]    + //BA
			img[0][ (i)* 	img[1][0] +(j+sx) ]    + //BB
			img[0][ (i+sy)*	img[1][0] +(j)	  ]	   - //BC
			img[0][ (i+sy)* img[1][0] +(j+sx) ]		 //BD
		)	

	//filtro blanco
	+	(
			img[0][  (i+sy)*img[1][0] +(j)	  ]	   + //BC
			img[0][  (i+sy)*img[1][0] +(j+sx) ]	   + //BD
			img[0][(i+sy*2)*img[1][0] +(j)	  ]	   - //BE
			img[0][(i+sy*2)*img[1][0] +(j+sx) ]		 //BF
		)	

	//filtro negro
	-	(
			img[0][(i+sy*2)*img[1][0] +(j)	  ]	   + //BE
			img[0][(i+sy*2)*img[1][0] +(j+sx) ]	   + //BF
			img[0][(i+sy*3)*img[1][0] +(j)	  ]	   - //BE
			img[0][(i+sy*3)*img[1][0] +(j+sx) ]		 //BF
		)	
	);

}).setDynamicOutput(true);

_kernel.HaarLineHor = Math.GPU.createKernel( function(img,sx,sy){

	const i = this.thread.y;
	const j = this.thread.x;
	
	return(
	//filtro negro
	-	(
			img[0][ (i)* 	img[1][0] +(j) 	  ]    + //BA
			img[0][ (i)* 	img[1][0] +(j+sx) ]    + //BB
			img[0][ (i+sy)*	img[1][0] +(j)	  ]	   - //BC
			img[0][ (i+sy)* img[1][0] +(j+sx) ]		 //BD
		)	

	//filtro blanco
	+	(
			img[0][ (i)*   img[1][0] +(j+sx)  ]    + //BB
			img[0][ (i)   *img[1][0] +(j+sx*2)]	   + //BE
			img[0][ (i+sy)*img[1][0] +(j+sx)  ]	   - //BD
			img[0][ (i+sy)*img[1][0] +(j+sx*2)]	     //BF
		)	

	//filtro blanco
	-	(
			img[0][ (i)   *img[1][0] +(j+sx*2)]	   + //BE
			img[0][ (i+sy)*img[1][0] +(j+sx*2)]	   + //BF
			img[0][ (i)   *img[1][0] +(j+sx*3)]	   - //BG
			img[0][ (i+sy)*img[1][0] +(j+sx*3)]	     //BH
		)		
	);

}).setDynamicOutput(true);

_kernel.HaarCross = Math.GPU.createKernel( function(img,sx,sy){

	const i = this.thread.y;
	const j = this.thread.x;

	return(
	//filtro negro
	-	(
			img[0][ (i)* 	img[1][0] +(j) 	  ]    + //BA
			img[0][ (i)* 	img[1][0] +(j+sx) ]    + //BB
			img[0][ (i+sy)*	img[1][0] +(j)	  ]	   - //BC
			img[0][ (i+sy)* img[1][0] +(j+sx) ]		 //BD
		)

	//filtro blanco
	+	(
			img[0][  (i+sy)*img[1][0] +(j)	  ]	   + //BC
			img[0][  (i+sy)*img[1][0] +(j+sx) ]	   + //BD
			img[0][(i+sy*2)*img[1][0] +(j)	  ]	   - //BE
			img[0][(i+sy*2)*img[1][0] +(j+sx) ]		 //BF
		)	
		
	//filtro negro
	-	(
			img[0][ (i)* 	img[1][0] +(j+sx) ]    + //BB
			img[0][ (i+sy)* img[1][0] +(j+sx) ]	   + //BD
			img[0][ (i)* 	img[1][0] +(j+sx*2)]   - //BE
			img[0][ (i+sy)* img[1][0] +(j+sx*2)]	 //BF
		)	

	//filtro blanco
	+	(
			img[0][  (i+sy)*img[1][0] +(j+sx) ]	   + //BD
			img[0][(i+sy*2)*img[1][0] +(j+sx) ]	   + //BF
			img[0][  (i+sy)*img[1][0] +(j+sx*2)]   - //BD
			img[0][(i+sy*2)*img[1][0] +(j+sx*2)]	 //BF
		)	
	);

}).setDynamicOutput(true);

_kernel.HaarOutline = Math.GPU.createKernel( function(img,sx,sy){

	const i = this.thread.y;
	const j = this.thread.x;

	return(
	//filtro negro ###
	-	(
			img[0][ (i)* 	img[1][0] +(j) 	  ]    + //BA
			img[0][ (i)* 	img[1][0] +(j+sx) ]    + //BB
			img[0][ (i+sy)*	img[1][0] +(j)	  ]	   - //BC
			img[0][ (i+sy)* img[1][0] +(j+sx) ]		 //BD
		)	

	//filtro blanco
	-	(
			img[0][  (i+sy)*img[1][0] +(j)	  ]	   + //BC
			img[0][  (i+sy)*img[1][0] +(j+sx) ]	   + //BD
			img[0][(i+sy*2)*img[1][0] +(j)	  ]	   - //BE
			img[0][(i+sy*2)*img[1][0] +(j+sx) ]		 //BF
		)	

	//filtro blanco
	-	(
			img[0][(i+sy*2)*img[1][0] +(j)	  ]	   + //BE
			img[0][(i+sy*2)*img[1][0] +(j+sx) ]	   + //BF
			img[0][(i+sy*3)*img[1][0] +(j)	  ]	   - //BE
			img[0][(i+sy*3)*img[1][0] +(j+sx) ]		 //BF
		)	
		
	//filtro negro ###
	-	(
			img[0][ (i)* 	img[1][0] +(j+sx) ]    + //BB
			img[0][ (i)* 	img[1][0] +(j+sx*2)]   + //BE
			img[0][ (i+sy)* img[1][0] +(j+sx) ]	   - //BD
			img[0][ (i+sy)* img[1][0] +(j+sx*2)]	 //BF
		)	
		
	//filtro negro
	+	(
			img[0][ (i+sy)* img[1][0] +(j+sx) ]	   - //BD
			img[0][ (i+sy)* img[1][0] +(j+sx*2)]   - //BF
			img[0][(i+sy*2)*img[1][0] +(j+sx) ]	   + //BD
			img[0][(i+sy*2)*img[1][0] +(j+sx*2)]	 //BF
		)	

	//filtro blanco
	-	(
			img[0][(i+sy*2)*img[1][0] +(j+sx) ]	   + //BD
			img[0][(i+sy*2)*img[1][0] +(j+sx*2)]   + //BF
			img[0][(i+sy*3)*img[1][0] +(j+sx) ]	   - //BD
			img[0][(i+sy*3)*img[1][0] +(j+sx*2)]	 //BF
		)		
		
	//filtro negro ###
	-	(
			img[0][ (i)* 	img[1][0] +(j+sx*2)]   + //BB
			img[0][ (i)* 	img[1][0] +(j+sx*3)]   + //BE
			img[0][ (i+sy)* img[1][0] +(j+sx*2)]   - //BD
			img[0][ (i+sy)* img[1][0] +(j+sx*3)]	 //BF
		)	
		
	//filtro negro
	-	(
			img[0][ (i+sy)* img[1][0] +(j+sx*2)]   + //BD
			img[0][ (i+sy)* img[1][0] +(j+sx*3)]   + //BF
			img[0][(i+sy*2)*img[1][0] +(j+sx*2)]   - //BD
			img[0][(i+sy*2)*img[1][0] +(j+sx*3)]	 //BF
		)	

	//filtro blanco
	-	(
			img[0][(i+sy*2)*img[1][0] +(j+sx*2)]   + //BD
			img[0][(i+sy*2)*img[1][0] +(j+sx*3)]   + //BF
			img[0][(i+sy*3)*img[1][0] +(j+sx*2)]   - //BD
			img[0][(i+sy*3)*img[1][0] +(j+sx*3)]	 //BF
		)	
	);

}).setDynamicOutput(true);

// TODO: Image Kernels #######################################################//
_kernel.flip = Math.GPU.createKernel( function(A){
	return A[0][ A[1][2] - 1 - this.thread.x ];
}).setDynamicOutput(true);

_kernel.padding = Math.GPU.createKernel( function(A,p){
	const y = this.thread.y;
	const x = this.thread.x;
	var val = 0;

	if( (p<=y && y<A[1][1]+p) && (p<=x && x<A[1][0]+p) ){
		var pos = ( (y-p) * A[1][0])+(x-p);
		val = A[0][pos];
	}	return val; 
}).setDynamicOutput(true);

_kernel.threshold = Math.GPU.createKernel( function(A,B){ 

	var th = 0;
	const x = this.thread.x;
	const y = this.thread.y;
	const pos = y * A[1][0] + x;

	if( A[0][pos]>B ) th=1; 
	else th=0; 
	return th;
}).setDynamicOutput(true);

_kernel.differenceEdgeDetection = Math.GPU.createKernel( function(A){
	var x = this.thread.x;
	var y = this.thread.y;
	var sum=0; var res=0; 
		
	for( var k=0; k<2; k++ ){ for( var l=0; l<2; l++ ){
		if(k==1 && l==1) continue;

		var pos = [
			(y+k) * A[1][0] + (x+l),
			(y-k+2) * A[1][0] + (x-l+2),
		];

		sum=A[0][pos[0]] - A[0][pos[1]];
		if( sum > res ) res = sum;
	}}	return res; 
}).setDynamicOutput(true);

_kernel.localBinaryPattern = Math.GPU.createKernel( function(A){

	var pos = [0,0];
	var x = this.thread.x;
	var y = this.thread.y;
	var sum=0; var index = 0;
		
	for( var k=0; k<3; k++ ){ for( var l=0; l<3; l++ ){
		if(k==1 && l==1) continue;

		pos[0] = (y+k) * A[1][0] + (x+l);
		pos[1] = (y+1) * A[1][0] + (x+1);

		if( A[0][pos[0]] > A[0][pos[1]] ) 
			sum += Math.pow(2,index); 
			index++;
	}}	return sum; 
}).setDynamicOutput(true);

_kernel.templateMatching = Math.GPU.createKernel( function(A,B,s){ 
		
	var a=0; var b=0; var c=0;
	const j = this.thread.x;
	const i = this.thread.y; 
 	var pos=[0,0];
	var acum = 0;

	for(var k=0; k<B[1][1] ;k++){for(var l=0; l<B[1][0] ;l++){
 			
		pos[0] = k * B[1][0] + l; 
		pos[1] = (k+i) * A[1][0] + (j+l); 
				
		a += B[0][pos[0]]*A[0][pos[1]];
		c += Math.pow(A[0][pos[1]],2);
		b += Math.pow(B[0][pos[0]],2);
						
	}}	acum = Math.sqrt( b * c )
	if( acum>0 ) acum = a/acum;
	return acum;
}).setDynamicOutput(true);

_kernel.convolution = Math.GPU.createKernel( function(A,B){ 
		
	const j = this.thread.x;
	const i = this.thread.y; 
	var pos = [0,0];
	var acum = 0;

	for( var k=0; k<B[1][1]; k++ ){ for( var l=0; l<B[1][0]; l++ ){
		
		pos[1] = (i+k) * A[1][0] + (j+l);
		pos[0] = k * B[1][0] + l;

		acum += A[0][pos[1]] * B[0][pos[0]];
	}}	return acum;
}).setDynamicOutput(true);

_kernel.correlation = Math.GPU.createKernel( function(A,B){ 
		
	const j = this.thread.x;
	const i = this.thread.y;
	var pos = [0,0]; 
	var acum = 0;

	for( var k=0; k<B[1][1]; k++ ){ for( var l=0; l<B[1][0]; l++ ){
			
		pos[1] = (i+k) * A[1][0] + (j+l);
		pos[0] = k * B[1][0] + l;

		acum += A[0][pos[1]] * B[0][pos[0]];
	}}	return acum;
}).setDynamicOutput(true);

_kernel.getImage_color = Math.GPU.createKernel( function(image){ 

	var value = 0;
	const j = this.thread.x;
	const i = this.thread.y; 
	const pixel = image[i][j];
	
	return pixel;
}).setDynamicOutput(true);

_kernel.getImage_gray = Math.GPU.createKernel( function(image){ 

	const j = this.thread.x;
	const i = this.thread.y; 
    const pixel = image[i][j];
	return ( pixel[0] + pixel[1] + pixel[2] + pixel[3] ) / 4;
	
}).setDynamicOutput(true);

_kernel.showImage_color = Math.GPU.createKernel( function(A,B,C,D,size){ 

	var pixel = [0,0,0,0] 
	const j = this.thread.x;
	const i = this.thread.y;
	const pos = i * size[0] + j;
	
	pixel[0] = A[ pos ];
	pixel[1] = B[ pos ];
	pixel[2] = C[ pos ];
	pixel[3] = D[ pos ];
	
	this.color( pixel[0],pixel[1],pixel[2],pixel[3] );

})	.setDynamicOutput(true)
	.setGraphical(true);

_kernel.showImage_gray = Math.GPU.createKernel( function(A){ 

	const j = this.thread.x;
	const i = this.thread.y; 
	const pixel = A[0][ i * A[1][0] + j ];
	this.color( pixel, pixel, pixel, pixel );

})	.setDynamicOutput(true)
	.setGraphical(true);	

_kernel.colorSegmentation = Math.GPU.createKernel( function(A){

	const color = [
		[0.75,0.75,0.75,1],
		[0.5,0.5,0.5,1],
		[0.5,0,0.5,1],
		[0.5,0.5,0,1],
		[0,0.5,0.5,1],
		[0.5,0,0,1],
		[0,0.5,0,1],
		[0,0,0.5,1],
		[0,0,0.5,1],
		[1,0,1,1],
		[1,1,0,1],
		[1,1,1,1],
		[0,0,0,1],
		[0,1,0,1],
		[0,1,1,0],
		[1,0,0,1]
	];
	
	var pixel1 = A[ this.thread.x ][0];
	var pixel2 = A[ this.thread.x ][1];
	var pixel3 = A[ this.thread.x ][2];
	var pixel4 = A[ this.thread.x ][3];
	
	
	return color[0];
});



