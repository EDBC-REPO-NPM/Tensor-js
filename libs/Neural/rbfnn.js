class _rbfnn{

	/// neural network ###################################### Legend ///
		//Array[0] = Loss
		//Array[1] = Bias
		//Array[2] = Weight
		//Array[3] = Output
		//Array[4] = Loss_Weight
		//Array[5] = Loss_Bias

	constructor( n_learning_rate=0.3, n_optimization=sgd, n_cost_type=mean_squared ){
	/// Variables ###################################### layer state ///
		this.i_layer = 0;
		this.topology_size = 3;
		this.cost = n_cost_type;
						
	/// Variables ################################### layer features ///
		this.type = new Array(this.topology_size);
		this.kernel = new Array(this.topology_size);
		this.topology = new Array(this.topology_size);
				
	/// Variables #################################### learning rate ///
		this.acceleration_rate = 0.9;
		this.optimization = n_optimization;
		this.learning_rate = n_learning_rate;

	/// layer #######################################################///
		this.layer = new Array( this.topology_size-1 );
		for( var i=this.topology_size-1; i--; ){
			this.layer[i] = new Array(6); }
		this.output; this.input;
	}

	/// neural network ########################### compile functions ///
	addLayer( n_neuron, k_neuron , n_type ){
		if( this.i_layer>0 ){ this.kernel[ this.i_layer-1 ] = k_neuron; }
		this.topology[ this.i_layer ] = n_neuron;
		this.type[ this.i_layer ] = n_type;
		this.i_layer += 1;
	}
				
	compile(){
		if( this.topology_size < this.i_layer ){
			console.log(`maximo de capas superado ${this.i_layer-this.topology_size} mlp`);
			process.exit(1);
		}	else if( this.topology_size > this.i_layer ){
			console.log(`capas faltantes ${this.topology_size-this.i_layer} mlp`);
			process.exit(1);
		}

		this.output = new Math.tensor(1,this.topology[this.topology_size-1]);
		this.input = new Math.tensor(1,this.topology[0]);
					
		for(var i=1; i<this.topology_size ;i++){
			this.layer[i-1][5] = new Math.tensor(1,this.topology[i]);
			this.layer[i-1][1] = new Math.tensor(1,this.topology[i]);
			this.layer[i-1][1].setOne();
			this.layer[i-1][5].setOne();
		}
	
		for(var i=0; i<this.topology_size-1 ;i++){ 
			this.layer[i][4] = new Math.tensor(this.topology[i],this.topology[i+1]);
			this.layer[i][2] = new Math.tensor(this.topology[i],this.topology[i+1]);
			this.layer[i][2].setRandom();
			this.layer[i][4].setOne();
		}
	}
			
	/// neural network ########################### traning functions ///
	forwrd(){
		for( var i=0; i<this.topology_size-1; i++ ){
			switch( this.type[i] ){
			case input_layer: 
				this.layer[i][3] = Math.gauss( Math.Tmul(this.layer[i][2],this.input),this.layer[i][1] ,true); 
			break;
			default: 
				this.layer[i][3] = Math.act( Math.Tadd(Math.Tmul(this.layer[i][2],this.layer[i-1][3]), this.layer[i][1]) ,this.kernel[i],true); 
			break;
		}}
	}
				
	/// neural network ################ Gradien descent with momentum ///
	backwrd(){
		switch( this.optimization ){
			case momentum:	
				for( var i=this.topology_size-1; i--; ){
					
					this.layer[i][5] = Math.Smul( this.layer[i][5], this.acceleration_rate );
					this.layer[i][4] = Math.Smul( this.layer[i][4], this.acceleration_rate );
					
					if( this.type[i+1]==output_layer ){
						this.layer[i][0] = Math.Tdot( Math.act(this.layer[i][3],this.kernel[i],false), Math.cost(this.output,this.layer[i][3],this.cost) );
						this.layer[i][4] = Math.Tadd( this.layer[i][4],Math.Tmul(this.layer[i][0],Math.Tsp(this.layer[i-1][3])) );	
						this.layer[i][5] = Math.Tadd( this.layer[i][5],this.layer[i][0] );
					}
					else{
						this.layer[i][0] = Math.Tmul( Math.Tsp(this.layer[i+1][2]), this.layer[i+1][0] );							
						this.layer[i][5] = Math.Tadd( this.layer[i][5],Math.Tdot(this.layer[i][0], Math.gauss(this.input,this.layer[i][1],4,false)) );
					}
						
					this.layer[i][2] = Math.Tadd( this.layer[i][2],Math.Smul(this.layer[i][4],this.learning_rate) );
					this.layer[i][1] = Math.Tadd( this.layer[i][1],Math.Smul(this.layer[i][5],this.learning_rate) );
				}
			break;
						
			default:	
				for( var i=this.topology_size-1; i--; ){
					if( this.type[i+1]==output_layer ){
						this.layer[i][0] = Math.Tdot( Math.act(this.layer[i][3],this.kernel[i],false), Math.cost(this.output,this.layer[i][3],this.cost) );
						this.layer[i][2] = Math.Tadd( this.layer[i][2], Math.Tmul(this.layer[i][0], Math.Smul(Math.Tsp(this.layer[i-1][3]),this.learning_rate)) );
						this.layer[i][1] = Math.Tadd( this.layer[i][1],Math.Smul(this.layer[i][0],this.learning_rate) );
					}	
					else{
						this.layer[i][0] = Math.Tmul( Math.Tsp(this.layer[i+1][2]),this.layer[i+1][0] );
						this.layer[i][1] = Math.Tadd( this.layer[i][1],Math.Tdot(this.layer[i][0], Math.Smul(Math.gauss(this.input,this.layer[i][1],4,false),this.learning_rate)) );
					}
				}
			break;
		}}
	
	/// neural network ############################# prvar functions ///

	getLoss(){ return Math.Tmul(Math.Tsp(this.layer[0][2]),this.layer[0][0]); }

	showOutput(){ this.layer[this.i_layer-2][3].showTensor(); }

	getOutput(){ return this.layer[this.i_layer-2][3]; }

	showModel(){
		for(var i=0; i<this.i_layer-1; i++){
			this.layer[i][1].showTensor();
			this.layer[i][2].showTensor();
		}
	}
			
	fit(X,Y){ this.input.setArray(X); this.output.setArray(Y); this.forwrd(); this.backwrd(); }
				
	predict(X){ this.input.setArray(X); this.forwrd(); }
	
	setCenter(A){ this.layer[0][1].setArray(A); }
		
} Math.RBFNN = _rbfnn;
