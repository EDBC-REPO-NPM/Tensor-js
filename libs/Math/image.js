
sobelEdge 			=0;
gaussFilter			=1;
prewittEdge 		=2;
sobelVertical 		=3;
sobelHorizontal 	=4;
robertsVertical 	=5;
prewittVertical 	=6;
robertsHorizontal 	=7;
prewittHorizontal 	=8;	

Math.getFilter = ( type )=>{
	
	var filter = [
		[ 1, 1, 1, 1, 1, 1, 1, 1, 1], //gauss		0
		[-1,-2,-1, 0, 0, 0, 1, 2, 1], //sobel   hor 1
		[-1, 0, 1,-2, 0, 2,-1, 0, 1], //sobel   ver	2
		[ 1, 0, 0,-1, 0, 0, 0, 0, 0], //robert  hor 3
		[ 0, 1,-1, 0, 0, 0, 0, 0, 0], //robert  ver	4
		[-1,-1,-1, 0, 0, 0, 1, 1, 1], //prewitt hor 5
		[-1, 0, 1,-1, 0, 1,-1, 0, 1], //prewitt ver 6
		[ 0,-1, 0,-1, 4,-1, 0,-1, 0], //sobel   edg 7
		[-1,-1,-1,-1, 8,-1,-1,-1,-1]  //prewitt edg 8
	];	var size=0; var index=0;

	switch( type ){
		case sobelEdge: 		size=3; index=7; break;
		case gaussFilter: 		size=3; index=0; break;
		case prewittEdge: 		size=3; index=8; break;
		case sobelVertical: 	size=3; index=2; break;
		case sobelHorizontal: 	size=3; index=1; break;
		case robertsVertical: 	size=2; index=4; break;
		case prewittVertical: 	size=3; index=6; break;
		case robertsHorizontal: size=2; index=3; break;
		case prewittHorizontal: size=3; index=5; break;
	} 	var C = new Math.tensor(size,size);
			C.setArray( filter[index] );
			return C;
}

Math.flip = (A)=>{
	var C = new Math.tensor(A.size[0],A.size[1]);
	if( C.size[2] > 1000 ){
		_kernel.flip.setOutput([A.size[2]]);
		C.data = _kernel.flip([A.data,A.size]); 
	} else { for( var i in C.data ){
		C.data[i] = A.data[ A.size[2]-1-i ];
	}}	return C;
}
	
Math.padding = (A,p=1)=>{
	var C = new Math.tensor( 2*p+A.size[0],2*p+A.size[1] );
	_kernel.padding.setOutput([C.size[0],C.size[1]]);
	C.data = _kernel.padding([A.data,A.size],p).map(x=>{return Array.from(x)}).flat();
	return C; 
}

Math.threshold = (A,B)=>{
	var C = new Math.tensor(A.size[0], A.size[1]);
	if( C.size[2] > 1000 ){
		
		_kernel.threshold.setOutput([C.size[2]]);
		C.data = _kernel.threshold([A.data,A.size],B);
		
	} else { for( var i in C.data ){
		if( A.data[i] > B ) C.data[i] = 1;
		else C.data[i] = 0;
	}}	return C;
}

Math.integralImage = (A)=>{
	var C = new Math.tensor( A.size[0], A.size[1] );
	for( var i=0; i<C.size[1]; i++){ for( var j=0; j<C.size[0]; j++){
		if(i==0 && j!=0){ C.data[C.getIndex(j,i)] = A.getValue(j,i) + A.getValue(j-1,i); }
		else if(i!=0 && j==0){ C.data[C.getIndex(j,i)] = A.getValue(j,i) + A.getValue(j,i-1); }
		else if(i!=0 && j!=0){ C.data[C.getIndex(j,i)] = A.getValue(j,i) + A.getValue(j,i-1) + A.getValue(j-1,i) - A.getValue(j-1,i-1); }
	}}	return C;
}

Math.differenceEdgeDetection = (A)=>{
	var C = new Math.tensor( A.size[0]-3+1, A.size[1]-3+1 );
	_kernel.differenceEdgeDetection.setOutput([C.size[0],C.size[1]]);
	C.data = _kernel.differenceEdgeDetection([A.data,A.size]).map(x=>{return Array.from(x)}).flat();
	return C;
}

Math.localBinaryPattern = (A)=>{
	var C = new Math.tensor( A.size[0]-2, A.size[1]-2 );
	_kernel.localBinaryPattern.setOutput([C.size[0],C.size[1]]);
	C.data = _kernel.localBinaryPattern([A.data,A.size]).map(x=>{return Array.from(x)}).flat();
	return C;
} 

Math.templateMatching = (A,B,s=0.1)=>{
	var C = new Math.tensor( (A.size[0]-B.size[0]+1), (A.size[1]-B.size[1]+1) );
	_kernel.templateMatching.setOutput([C.size[0],C.size[1]]);
	C.data = _kernel.templateMatching([A.data,A.size],[B.data,B.size],s).map(x=>{return Array.from(x)}).flat();
	return C;
}

Math.convolution = (A,B)=>{ var kernel;
	var C = new Math.tensor(A.size[0]-B.size[0]+1, A.size[1]-B.size[1]+1);
	if(A.size[2] < B.size[2]){
		console.log(`convolucion: A:${A.size[0]}x${A.size[1]} B:${B.size[0]}x${B.size[1]}`);
		process.exit(1);
	}	B = Math.flip(B);
	_kernel.convolution.setDynamicOutput(true);
	_kernel.convolution.setOutput([C.size[0],C.size[1]]);
	C.data = _kernel.convolution([A.data,A.size],[B.data,B.size]).map(x=>{return Array.from(x)}).flat();
	return C;
}

Math.correlation = (A,B)=>{ var kernel;
	var C = new Math.tensor(A.size[0]-B.size[0]+1, A.size[1]-B.size[1]+1);
	if(A.size[2] < B.size[2]){
		console.log(`correlacion: A:${A.size[0]}x${A.size[1]} B:${B.size[0]}x${B.size[1]}`);
		process.exit(1);}
	_kernel.correlation.setOutput([C.size[0],C.size[1]]);
	C.data = _kernel.correlation([A.data,A.size],[B.data,B.size]).map(x=>{return Array.from(x)}).flat();
	return C;
}

Math.HaarEdgeVer = ( img, sx=1, sy=1)=>{
	img = Math.integralImage(img); 
	var C = new Math.tensor( img.size[0]-sx-1,img.size[1]-(sy*2)-1 );

	_kernel.HaarEdgeVer.setOutput([C.size[0],C.size[1]]);
	C.data = _kernel.HaarEdgeVer([img.data,img.size],sx,sy).map(x=>{return Array.from(x)}).flat();
	return C;
}

Math.HaarEdgeHor = ( img, sx=1, sy=1)=>{
	img = Math.integralImage(img);
	var C = new Math.tensor( img.size[0]-(sx*2),img.size[1]-sy ); 

	_kernel.HaarEdgeHor.setOutput([C.size[0],C.size[1]]);
	C.data = _kernel.HaarEdgeHor([img.data,img.size],sx,sy).map(x=>{return Array.from(x)}).flat();
	return C;
}

Math.HaarLineVer = ( img, sx=1, sy=1)=>{
	img = Math.integralImage(img);
	var C = new Math.tensor( img.size[0]-sx,img.size[1]-(sy*3) ); 

	_kernel.HaarLineVer.setOutput([C.size[0],C.size[1]]);
	C.data = _kernel.HaarLineVer([img.data,img.size],sx,sy).map(x=>{return Array.from(x)}).flat();
	return C;
}

Math.HaarLineHor = ( img, sx=1, sy=1)=>{
	img = Math.integralImage(img);
	var C = new Math.tensor( img.size[0]-(sx*3),img.size[1]-sy );

	_kernel.HaarLineHor.setOutput([C.size[0],C.size[1]]);
	C.data = _kernel.HaarLineHor([img.data,img.size],sx,sy).map(x=>{return Array.from(x)}).flat();
	return C;
}

Math.HaarCross = ( img, sx=1, sy=1)=>{
	img = Math.integralImage(img);
	var C = new Math.tensor( img.size[0]-(sx*2),img.size[1]-(sy*2) ); 

	_kernel.HaarCross.setOutput([C.size[0],C.size[1]]);
	C.data = _kernel.HaarCross([img.data,img.size],sx,sy).map(x=>{return Array.from(x)}).flat();
	return C;
}

Math.HaarOutline = ( img, sx=1, sy=1)=>{
	img = Math.integralImage(img);
	var C = new Math.tensor( img.size[0]-(sx*3),img.size[1]-(sy*3) ); 

	_kernel.HaarOutline.setOutput([C.size[0],C.size[1]]);
	C.data = _kernel.HaarOutline([img.data,img.size],sx,sy).map(x=>{return Array.from(x)}).flat();
	return C;
}

Math.getImage = (image,size,gray=true)=>{
	var C = new Math.tensor(size[0],size[1]);
	if( !gray ){ 
		var C = new Math.tensor(size[0],size[1]);
		_kernel.getImage_color.setOutput([size[0],size[1]]);
		C.data = _kernel.getImage_color(image).map(x=>{return Array.from(x)}).flat();	
		C.data = [ C.data.map(x=>x[0]),C.data.map(x=>x[1]),C.data.map(x=>x[2]),C.data.map(x=>x[3]) ];
	} else {
		_kernel.getImage_gray.setOutput([size[0],size[1]]);
		C.data = _kernel.getImage_gray(image).map(x=>{return Array.from(x)}).flat();
	}	return C;
}

Math.showImage = ( A )=>{
	if( A.data.length<A.size[2] ){
		_kernel.showImage_color.setOutput([A.size[0],A.size[1]])
		_kernel.showImage_color(A.data[0],A.data[1],A.data[2],A.data[3],A.size);
	} else {
		_kernel.showImage_gray.setOutput([A.size[0],A.size[1]])
		_kernel.showImage_gray([A.data,A.size]);
	}
}

Math.colorSegmentation = ( A )=>{
	var C = new tensor(A.size[0],A.size[1]);
}


/*	
	tensor maxpooling(tensor A){ //FIXME: to Optimize
		I_reg[3].relloc(A.size[0]/2,A.size[1]/2);
			
		for(int i=0,n=0; n<I_reg[3].size[1] ;n++){for(int j=0,m=0; m<I_reg[3].size[0] ;m++){ 
			for(int k=2; k--;){for(int l=2; l--;){
				if(I_reg[3].data[I_reg[3].get_index(n,m)] < A.data[A.get_index(i+k,j+l)])
					I_reg[3].data[I_reg[3].get_index(n,m)] = A.data[A.get_index(i+k,j+l)];
					
		}}j+=2;}i+=2;}
	return I_reg[3];}
	
	tensor dmaxpooling(tensor A, tensor B){ //FIXME: to Optimize
		I_reg[3].relloc(A.size[0],A.size[1]);
			
		for(int i=0,n=0; n<B.size[1] ;n++){for(int j=0,m=0; m<B.size[0] ;m++){	
			float acum=0; int indx=0;	

			for(int k=2; k--;){for(int l=2; l--;){
				if(acum < A.data[A.get_index(i+k,j+l)]){
					acum = A.data[A.get_index(i+k,j+l)];
					indx = I_reg[3].get_index(i+k,j+l);
				}}}
					
			I_reg[3].data[indx] = B.data[B.get_index(n,m)];

		j+=2;}i+=2;}				
	return I_reg[3];}
		

tensor labelling(tensor A){
		I_reg[1].relloc(A.size[0],A.size[1]);
		int label=1;
	
		for(int i=I_reg[1].size[1]-2; i--;){for(int j=I_reg[1].size[0]-2; j--;){
			if(A.data[A.get_index(i,j)] != 0){
				for(int k=3; k--;){ for(int l=3; l--;){
					if(I_reg[1].data[I_reg[1].get_index(i+k,j+l)] != 0){
						I_reg[1].data[I_reg[1].get_index(i+1,j+1)] = I_reg[1].data[I_reg[1].get_index(i+k,j+l)];
				}}}
				if(I_reg[1].data[I_reg[1].get_index(i+1,j+1)] == 0){ I_reg[1].data[I_reg[1].get_index(i+1,j+1)]=label; label++; }
		}}}
	
		I_reg[2].relloc(4,label-1);
		for(int i=I_reg[1].size[1], lb=0; i--;){for(int j=I_reg[1].size[0]; j--;){
			if((lb=I_reg[1].data[I_reg[1].get_index(i,j)]) != 0){
				if( I_reg[2].data[I_reg[2].get_index(lb-1,0)] == 0 || I_reg[2].data[I_reg[2].get_index(lb-1,0)] > j){ I_reg[2].data[I_reg[2].get_index(lb-1,0)] = j; }
				if(	I_reg[2].data[I_reg[2].get_index(lb-1,1)] == 0 || I_reg[2].data[I_reg[2].get_index(lb-1,1)] > i){ I_reg[2].data[I_reg[2].get_index(lb-1,1)] = i; }
				if( I_reg[2].data[I_reg[2].get_index(lb-1,2)] < j ){ I_reg[2].data[I_reg[2].get_index(lb-1,2)] = j;}
				if( I_reg[2].data[I_reg[2].get_index(lb-1,3)] < i ){ I_reg[2].data[I_reg[2].get_index(lb-1,3)] = i;}
		}}}

		for(int i=I_reg[2].size[1]; i--;){
			I_reg[2].data[I_reg[2].get_index(i,0)] /= I_reg[1].size[0];
			I_reg[2].data[I_reg[2].get_index(i,1)] /= I_reg[1].size[1];
			I_reg[2].data[I_reg[2].get_index(i,2)] /= I_reg[1].size[0];
			I_reg[2].data[I_reg[2].get_index(i,3)] /= I_reg[1].size[1];}

	return I_reg[2];}
	

*/