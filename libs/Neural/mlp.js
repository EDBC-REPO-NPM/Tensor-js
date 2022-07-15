// Optimization ##############################################################//
sgd			=0;
momentum	=1; 
	
// Layer Const ###############################################################//
input_layer	=0;
hidden_layer=1;
output_layer=2;

// Cost function kernel ######################################################//
mean_squared =0;
cross_entropy=1;

class _mlp{

	/// neural network ###################################### Legend ///
		//Array[0] = Loss
		//Array[1] = Bias
		//Array[2] = Weight
		//Array[3] = Output
		//Array[4] = Loss_Weight
		//Array[5] = Loss_Bias

	constructor( n_layer=3, n_learning_rate=0.3, n_optimization=sgd, n_cost_type=mean_squared ){
	/// Variables ###################################### layer state ///
		this.i_layer = 0;
		this.cost = n_cost_type;
		this.topology_size = n_layer;
						
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
			this.layer[i-1][1].setValue(-1);
			this.layer[i-1][5].setValue(-1);
		}
	
		for(var i=0; i<this.topology_size-1 ;i++){ 
			this.layer[i][4] = new Math.tensor(this.topology[i],this.topology[i+1]);
			this.layer[i][2] = new Math.tensor(this.topology[i],this.topology[i+1]);
			this.layer[i][2].setRandom(-1,2);
			this.layer[i][4].setValue(-1);
		}
	}

	/// neural network ########################### traning functions ///
	forwrd(){
		for(var i=0; i<this.topology_size-1; i++){
			switch( this.type[i] ){
				case input_layer:
					this.layer[i][3] = Math.act(Math.Tadd(Math.Tmul(this.layer[i][2],this.input),this.layer[i][1]),this.kernel[i],true); 
				break;

				default:
					this.layer[i][3] = Math.act(Math.Tadd(Math.Tmul(this.layer[i][2],this.layer[i-1][3]),this.layer[i][1]),this.kernel[i],true);
				break;
			}
		}
	}
				
	/// neural network ################ Gradien descent with momentum ///
	backwrd(){
		switch( this.optimization ){
			case momentum: 
				if( this.topology_size>2 ){
					for(var i=this.topology_size-1; i--;){
						this.layer[i][5] = Math.Smul( this.layer[i][5],this.acceleration_rate );
						this.layer[i][4] = Math.Smul( this.layer[i][4],this.acceleration_rate );
							
						if( this.type[i+1]==output_layer ){
							this.layer[i][0] = Math.Tdot( Math.act(this.layer[i][3],this.kernel[i],false), Math.cost(this.output, this.layer[i][3],this.cost) );
							this.layer[i][4] = Math.Tadd( this.layer[i][4], Math.Tmul( this.layer[i][0], Math.Tsp(this.layer[i-1][3]) ) );
						}
		
						else if( this.type[i]==hidden_layer){
							this.layer[i][0] = Math.Tdot( Math.act(this.layer[i][3],this.kernel[i],false), Math.Tmul( Math.Tsp(this.layer[i+1][2]),this.layer[i+1][0] ));
							this.layer[i][4] = Math.Tadd( this.layer[i][4], Math.Tmul( this.layer[i][0], Math.Tsp(this.layer[i-1][3]) ) );
						}
		
						else{
							this.layer[i][0] = Math.Tdot( Math.act(this.layer[i][3],this.kernel[i],false), Math.Tmul( Math.Tsp(this.layer[i+1][2]),this.layer[i+1][0] ));
							this.layer[i][4] = Math.Tadd( this.layer[i][4], Math.Tmul( this.layer[i][0], Math.Tsp(this.input) ) );
						}

						this.layer[i][5] = Math.Tadd( this.layer[i][5],this.layer[i][0] );
						this.layer[i][2] = Math.Tadd( this.layer[i][2], Math.Smul( this.layer[i][4],this.learning_rate ) );
						this.layer[i][1] = Math.Tadd( this.layer[i][1], Math.Smul( this.layer[i][5],this.learning_rate ) );
					}
				} else {
					this.layer[0][0] = Math.Tdot( Math.act(this.layer[0][3],this.kernel[0],false), Math.cost(this.output,this.layer[0][3],this.cost) );
					
					this.layer[0][5] = Math.Smul( this.layer[0][5], this.acceleration_rate );
					this.layer[0][4] = Math.Smul( this.layer[0][4], this.acceleration_rate );

					this.layer[0][5] = Math.Tadd( this.layer[0][5], this.layer[0][0] );
					this.layer[0][4] = Math.Tadd( this.layer[0][4], Math.Tmul( this.layer[0][0], Math.Tsp(this.input) ));
							
					this.layer[0][2] = Math.Tadd( this.layer[0][2], Math.Smul(this.layer[0][4],this.learning_rate) );
					this.layer[0][1] = Math.Tadd( this.layer[0][1], Math.Smul(this.layer[0][5],this.learning_rate) );
				}	break;
						
			default: 
				if( this.topology_size>2 ){
					for( var i=this.topology_size-1; i--; ){
		
						if( this.type[i+1]==output_layer ){
							this.layer[i][0] = Math.Tdot( Math.act(this.layer[i][3],this.kernel[i],false),Math.cost(this.output,this.layer[i][3],this.cost) );
							this.layer[i][2] = Math.Tadd( this.layer[i][2], Math.Tmul(this.layer[i][0], Math.Smul(Math.Tsp(this.layer[i-1][3]),this.learning_rate)) ); 
						}
		
						else if( this.type[i]==hidden_layer ){
							this.layer[i][0] = Math.Tdot(Math.act(this.layer[i][3],this.kernel[i],false), Math.Tmul(Math.Tsp(this.layer[i+1][2]),this.layer[i+1][0]));
							this.layer[i][2] = Math.Tadd( this.layer[i][2],Math.Tmul( this.layer[i][0], Math.Smul( Math.Tsp(this.layer[i-1][3]), this.learning_rate) ) );
						}
		
						else{
							this.layer[i][0] = Math.Tdot( Math.act(this.layer[i][3],this.kernel[i],false), Math.Tmul(Math.Tsp(this.layer[i+1][2]),this.layer[i+1][0]) );
							this.layer[i][2] = Math.Tadd( this.layer[i][2], Math.Tmul( this.layer[i][0], Math.Smul(Math.Tsp(this.input),this.learning_rate) ) );
						}
						
						this.layer[i][1] = Math.Tadd( this.layer[i][1], Math.Smul( this.layer[i][0], this.learning_rate ) );
					}

				} else {
					this.layer[0][0] = Math.Tdot( Math.act( this.layer[0][3], this.kernel[0], false), Math.cost( this.output, this.layer[0][3], this.cost) );
					this.layer[0][2] = Math.Tadd( this.layer[0][2],Math.Tmul( this.layer[0][0], Math.Smul( Math.Tsp(this.input),this.learning_rate ) ));
					this.layer[0][1] = Math.Tadd( this.layer[0][1],Math.Smul( this.layer[0][0], this.learning_rate ));
				}	break;
		}
	}

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

	/// neural network ############################# prvar functions ///
	fit(X,Y){
		this.input.setArray(X); this.output.setArray(Y);
		this.forwrd(); this.backwrd();
	}
				
	predict(X){ 
		this.input.setArray(X); this.forwrd(); 
	}

}	Math.MLP = _mlp;

