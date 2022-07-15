
// Activation function kernel ################################################//
relu         =0;
tangh        =1;
gauss 	     =7;
atang        =2;
linear       =3;
softmax      =4;
sigmoid      =5;
softplus     =6;

Math.cost = (Y,A,kernel)=>{
	var C = new Math.tensor(Y.size[0], Y.size[1]);
	if( C.size[2] > 1000 ){
		
		_kernel.cost.setOutput([Y.size[2]]);
		C.data = _kernel.cost(Y.data,A.data,kernel);
		
	} else { for( var i in C.data ){
		switch(kernel){
		case cross_entropy: C.data[i] = (Y.data[i]/A.data[i]) - (1-Y.data[i])/(1-A.data[i]); break;
		default: C.data[i] = Y.data[i] - A.data[i]; break; 
	}}} return C;
}
	
Math.linear = (X,dev=true)=>{
	var C = new Math.tensor(X.size[0], X.size[1]);
	if( C.size[2] > 1000 ){
	
		_kernel.linear.setOutput([X.size[2]]);
		C.data = _kernel.linear(X.data,dev);
		
	} else { for( var i in C.data ){
		if(dev == true) C.data[i] = X.data[i]; 
		else C.data[i] = 1; 
	}} 	return C;
}
		
Math.tangh = (X,dev=true)=>{ 
	var C = new Math.tensor(X.size[0], X.size[1]);
	if( X.size[2] > 1000 ){
		
		_kernel.tangh.setOutput([X.size[2]]);
		C.data = _kernel.tangh(X.data,dev);
		
	} else { for( var i in C.data ){
		if(dev == true) C.data[i]=Math.tanh( X.data[i] ); 
		else C.data[i]=1-( Math.pow(X.data[i],2) ); 
	}} return C;
}

Math.softPlus = (X,dev=true)=>{
	var C = new Math.tensor(X.size[0], X.size[1]);
	if( X.size[2] > 1000 ){
	
		_kernel.softPlus.setOutput([X.size[2]]);
		C.data = _kernel.softPlus(X.data,dev);
		
	} else{ for( var i in C.data ){
		if(dev == true) C.data[i]=Math.log(1+Math.exp(X.data[i])); 
		else C.data[i]=1/(1+Math.exp(-X.data[i])); 
	}}	return C;
}

Math.atang = (X,dev=true)=>{
	var C = new Math.tensor(X.size[0], X.size[1]);
	if( C.size[2] > 1000 ){
	
		_kernel.atang.setOutput([C.size[2]]);
		C.data = _kernel.atang(X.data,dev);

	} else { for( var i in C.data ){
		if(dev == true) C.data[i]=Math.atan(X.data[i]); 
		else C.data[i]=1/(1+Math.pow(X.data[i],2)); 
	}}	return C;
}

Math.sigmoid = (X,dev=true)=>{ 
	var C = new Math.tensor(X.size[0], X.size[1]);
	if( C.size[2] > 1000 ){
	
		_kernel.sigmoid.setOutput([X.size[2]]);
		C.data = _kernel.sigmoid(X.data,dev);
		
	} else { for( i in C.data ){
		if(dev == true) C.data[i] = 1/(1+Math.exp(-X.data[i])); 
		else C.data[i] = X.data[i]*(1-X.data[i]); 
	}} return C;
}

Math.binary = (X,dev=true)=>{ 
	var C = new Math.tensor(X.size[0], X.size[1]);
	if( C.size[2] > 1000 ){
	
		_kernel.sigmoid.setOutput([X.size[2]]);
		C.data = _kernel.binary(X.data,dev);
		
	} else { for( i in C.data ){
		if(dev == true){
			if( X.data[i]>1 ) C.data[i] = 1;
			else if( x<-1 ) C.data[i] = -1;
			else C.data[i] = X.data[i];			
		} else {
			if( -1 < X.data[i] && X.data[i] < 1 ) 
				C.data[i] = 1;
			else 
				C.data[i] = 0;			
		}
	}} return C;
}

Math.relu = (X,dev=true)=>{
	var C = new Math.tensor(X.size[0], X.size[1]);
	if( C.size[2] > 1000 ){
	
		_kernel.relu.setOutput([C.size[2]]);
		C.data = _kernel.relu(X.data,dev);
		
	} else for( var i in C.data ){
		if(dev == true){
			if(X.data[i]>0) C.data[i]=X.data[i];
			else C.data[i]=0;
		} else {
			if(X.data[i]>0) C.data[i]=1;
			else C.data[i]=0;
		}
	}	return C;
}

Math.softMax = (X,dev=true)=>{ var acum = 0;
	var C = new Math.tensor(X.size[0], X.size[1]);
	if( C.size[2] > 1000 ){
		if(dev==true){
			for( var i in X.data ){ acum += Math.exp(X.data[i]); }
			_kernel.softMax1.setOutput([X.size[2]]); 
			C.data = _kernel.softMax1(X.data,acum);
		} else {
			_kernel.softMax2.setOutput([X.size[2],X.size[2]]); 
			C.data = _kernel.softMax2(C.data,[X.data,X.size]);
			
			_kernel.softMax3.setOutput([X.size[2]]);
			C.data = _kernel.softMax3(C.data, X.data);
		}
	} else 	{
		for( var i in C.data ){
			if( dev==true ) acum += Math.exp(X.data[i]) 
			else for( var j in C.data ){
				if( i!=j ) C.data[i] += X.data[j]; 
			}
		}

		for( var i in C.data ){
			if( dev==true ) C.data[i] = Math.exp(X.data[i])/acum; 
			else C.data[i] *= X.data[i];
		}
	}	return C;
}

Math.gauss = (X,Y,T=0.4,dev=true)=>{
	var C = new Math.tensor(Y.size[0],Y.size[1]);
	if( C.size[2] > 1000 ){
	
		_kernel.gauss.setOutput([X.size[0],X.size[1]]);
		C.data =_kernel.gauss(X.data,Y.data,T,dev).map(x=>{return Array.from(x)}).flat();
		
	} else { for( var i in C.data ){ for( var j in C.data ){
		if(dev == true) C.data[i] = Math.exp(-Math.pow(X.data[j]-Y.data[i],2) / (2*Math.pow(T,2)));
		else C.data[i] = Math.exp(-Math.pow(X.data[j]-Y.data[i],2) / (2*Math.pow(T,2))) * ((X.data[j]-Y.data[i]) / (Math.pow(T,2))); 
	}}} return C;
}
	
Math.act = (X,kernel,dev=true)=>{
	switch(kernel){
		case relu:		return Math.relu(X,dev);	break;
		case atang:	 	return Math.atang(X,dev);	break;
		case tangh:	 	return Math.tangh(X,dev);	break;
		case linear:	return Math.linear(X,dev);  break;
		case sigmoid: 	return Math.sigmoid(X,dev); break;
		case softmax:	return Math.softMax(X,dev); break;
		case softplus:	return Math.softPlus(X,dev);break;
		default: 		return Math.sigmoid(X,dev); break;
	}
}