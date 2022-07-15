// TODO: Scalar, Matrix, Vector Detector ############################################ //

Math.isMatrix = ( A ) => { return A?.size[0] > 1 && A?.size[1] > 1 ? true : false; }
Math.isScalar = ( A ) => { return typeof A == 'number' || A?.size[2] == 1 ? true : false; }
Math.isVector = ( A ) => { return typeof A != 'number' && (A?.size[0] == 1 || A?.size[1] == 1) ? true : false; }

// TODO: Scalar, Matrix, Vector Detector ############################################ //
class _math extends _tensor { 

	Mult( ...Tensor ){
		for( var i=0; i<Tensor.length; i++ ){

		}
	}

	Add( ...Tensor ){
		for( var i=0; i<Tensor.length; i++ ){

		}
	}

	Sub( ...Tensor ){
		for( var i=0; i<Tensor.length; i++ ){

		}
	}

}

module.extends = _math;

// TODO: Multiplicacion de Un Tensor Con Un Escalar ############################################ //
Math.Mult = (A,B)=>{
	var  C = new Math.tensor(A.size[0], A.size[1]);
	C.data[i] = A.data[i] * B; 
	return C;
}

// TODO: Exponensial de un tensor ############################################################## //
Math.Texp = (A)=>{
	var C = new Math.tensor(A.size[0],A.size[1]);
	if( C.size[2] > 1000 ){
	
		_kernel.Texp.setOutput([A.size[2]]);
		C.data = _kernel.Texp(A.data);
	
	} else { for( var i in C.data ){
		C.data[i] = Math.exp( A.data[i] ); }	
	}	return C;
}
	
// TODO: Potencial de un tensor ##################################################################### //
Math.Tpow = (A,B)=>{
	var C = new Math.tensor(A.size[0],A.size[1]);
	if( C.size[2] > 1000 ){
	
		_kernel.Tpow.setOutput([A.size[2]]);
		C.data = _kernel.Tpow(A.data,B);
	
	} else { for( var i in C.data ){
		C.data[i] = Math.pow( A.data[i],B ); }	
	}	return C;
}
	
// TODO: Suma de un tensor ######################################################################### //
Math.Tadd = (A,B)=>{
	var C = new Math.tensor(A.size[0], A.size[1]);	
	if(A.size[0] != B.size[0] || A.size[1] != B.size[1]){
		console.log(`adicion erronea: A:${A.size[0]}x${A.size[1]} B:${B.size[0]}x${B.size[1]}`);
		process.exit(1);}
		
	if( C.size[2] > 1000 ){
		
		_kernel.Tadd.setOutput([A.size[2]]);
		C.data = _kernel.Tadd(A.data,B.data);
		
	} else { for( var i in C.data ){
		C.data[i] = A.data[i] + B.data[i]; }
	}	return C;
}
	
// TODO: Resta de un tensor ####################################################################### //
Math.Tsub = (A,B)=>{
	var C = new Math.tensor(A.size[0], A.size[1]);	
	if(A.size[0] != B.size[0] || A.size[1] != B.size[1]){
		console.log(`substraccion erronea: A:${A.size[0]}x${A.size[1]} B:${B.size[0]}x${B.size[1]}`);
		process.exit(1);}
		
	if( C.size[2] > 1000 ){
	
		_kernel.Tsub.setOutput([A.size[2]]);
		C.data = _kernel.Tsub(A.data,B.data);
	
	} else { for( var i in C.data ){
		C.data[i] = A.data[i] - B.data[i]; }
	}	return C;
}
	
// TODO: Multiplicacion * de un tensor ########################################################### //
Math.Tmul = (A,B)=>{
	var C = new Math.tensor(B.size[0],A.size[1]);
	if(A.size[0] != B.size[1]){
		console.log(`producto erroneo A:${A.size[0]}x${A.size[1]} B:${B.size[0]}x${B.size[1]}`);
		process.exit(1);}

	if( C.size[2] > 1000 ){
	
		_kernel.Tmul.setOutput([B.size[0],A.size[1]]);
		C.data = _kernel.Tmul([A.data,A.size],[B.data,B.size]).map(x=>{return Array.from(x)}).flat();
	
	} else {
		for( var i=0; i<A.size[1]; i++ ){ for( var j=0; j<B.size[0]; j++ ){
			for( var k=0; k<A.size[0]; k++ ){
				C.data[C.getIndex(j,i)] += A.getValue(k,i) * B.getValue(j,k);
			}
		}}	
	}	return C;
}
	
// TODO: Multiplicacion .* de un tensor ######################################################### //
Math.Tdot = (A,B)=>{
	var C = new Math.tensor(A.size[0], A.size[1]);	
	if(A.size[0] != B.size[0] || A.size[1] != B.size[1]){
		console.log(`producto lineal erronea: A:${A.size[0]}x${A.size[1]} B:${B.size[0]}x${B.size[1]}`);
		process.exit(1);}

	if( C.size[2] > 1000 ){
	
		_kernel.Tdot.setOutput([A.size[2]]);
		C.data = _kernel.Tdot(A.data,B.data);
	
	} else { for( var i in C.data ){
		C.data[i] = A.data[i] * B.data[i];
	}}	return C;
}


