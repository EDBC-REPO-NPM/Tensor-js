
///##########################################################################///
///										RNN								 	 ///
///##########################################################################///
class _rnn{
	
	/// neural network ########################### traning functions ///
		//Array[0] = weight_tmp
		//Array[1] = Loss
		//Array[2] = Bias
		//Array[3] = output_tmp
		//Array[4] = weight
		//Array[5] = Output
		//Array[6] = loss_weight
		//Array[7] = loss_bias
		//Array[8] = loss_weight_tmp
			
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
				this.layer[i] = new Array(9); }
			this.output; this.input;
	}
				
	/// neural network ########################### compile functions ///
	addLayer( n_neuron, k_neuron , n_type){
		if( this.i_layer>0 ){this.kernel[this.i_layer-1] = k_neuron;}
		this.topology[this.i_layer] = n_neuron;
		this.type[this.i_layer] = n_type;
		this.i_layer += 1;
	}
				
	compile(){
		if( this.topology_size < this.i_layer ){
			console.log(`maximo de capas superado ${this.i_layer - this.topology_size} RNN`);
			process.exit(1);
		} else if( this.topology_size > this.i_layer ){
			console.log(`capas faltantes ${this.topology_size-this.i_layer} RNN`);
			process.exit(1);
		}
			
		this.output = new Math.tensor(1,this.topology[this.topology_size-1]);
		this.input  = new Math.tensor(1,this.topology[0]);

		for(var i=1; i<this.topology_size ;i++){
			this.layer[i-1][8] = new Math.tensor(this.topology[i],this.topology[i]);
			this.layer[i-1][2] = new Math.tensor(1,this.topology[i]);
			this.layer[i-1][7] = new Math.tensor(1,this.topology[i]);
			this.layer[i-1][5] = new Math.tensor(1,this.topology[i]);
			this.layer[i-1][3] = new Math.tensor(1,this.topology[i]);
			this.layer[i-1][8].setValue(-1);
			this.layer[i-1][2].setValue(-1);
			this.layer[i-1][7].setValue(-1);
		}
						
		for(var i=0; i<this.topology_size-1 ;i++){
			this.layer[i][6] = new Math.tensor(this.topology[i],this.topology[i+1]);
			this.layer[i][4] = new Math.tensor(this.topology[i],this.topology[i+1]);
			this.layer[i][0] = new Math.tensor(this.topology[i],this.topology[i]);
			this.layer[i][0].setRandom(-1,2);
			this.layer[i][4].setRandom(-1,2);
			this.layer[i][6].setValue(-1);
		}
	}

	/// neural network ############################ FrontPropagation ///
		forwrd(){
			for(var i=0; i<this.topology_size-1; i++){
				switch( this.type[i] ){
					
				case hidden_layer:
					this.layer[i][5] = Math.Tadd( Math.Tmul( this.layer[i][4],this.layer[i-1][5] ), this.layer[i][2] );
					this.layer[i-1][3] = Math.Tmul( this.layer[i][0],this.layer[i-1][5] );
					this.layer[i][5] = Math.Tadd( this.layer[i][5],this.layer[i][3] );
				break;
					
				case input_layer:
					this.layer[i][5] = Math.Tadd( Math.Tmul( this.layer[i][4],this.input ) ,this.layer[i][2] );
					this.layer[i][5] = Math.Tadd( this.layer[i][5],this.layer[i][3] );
				break;
					
				default:
					this.layer[i][5] = Math.Tmul( this.layer[i][4], Math.Tadd(this.layer[i-1][5],this.layer[i][2]) );
				break; }
						
				this.layer[i][5] = Math.act( this.layer[i][5],this.kernel[i],true );
			}
		}

	/// neural network ############### Gradien descent with momentum ///
		backwrd(){
			switch( this.optimization ){
				case momentum:
					for( var i=this.topology_size-1; i--; ){
						
						this.layer[i][7] = Math.Smul( this.layer[i][7],this.acceleration_rate );
						this.layer[i][6] = Math.Smul( this.layer[i][6],this.acceleration_rate );
						this.layer[i][8] = Math.Smul( this.layer[i][8],this.acceleration_rate );
						
						if( this.type[i+1]==output_layer ){
							this.layer[i][1] = Math.Tdot( Math.act(this.layer[i][5],this.kernel[i],false), Math.cost(this.output,this.layer[i][5],this.cost) );
							this.layer[i][6] = Math.Tadd( this.layer[i][6], Math.Tmul(this.layer[i][1],Math.Tsp(this.layer[i-1][5])) );
						}
	
						else if( this.type[i]==hidden_layer ){
							this.layer[i][1] = Math.Tdot( Math.act(this.layer[i][5],this.kernel[i],false), Math.Tmul(Math.Tsp(this.layer[i+1][4]),this.layer[i+1][1]) );
							this.layer[i][6] = Math.Tadd( this.layer[i][6], Math.Tmul(this.layer[i][1], Math.Tsp(this.layer[i-1][5])) );
							this.layer[i][8] = Math.Tadd( this.layer[i][8], Math.Tmul(this.layer[i][1], Math.Tsp(this.layer[i][3])) );
							this.layer[i][0] = Math.Tadd( this.layer[i][0], Math.Smul(this.layer[i][8], this.learning_rate) );
						}
	
						else{
							this.layer[i][1] = Math.Tdot( Math.act( this.layer[i][5],this.kernel[i],false), Math.Tmul(Math.Tsp(this.layer[i+1][4]), this.layer[i+1][1]) );
							this.layer[i][6] = Math.Tadd( this.layer[i][6], Math.Tmul(this.layer[i][1],Math.Tsp(this.input)) );
						}
						
						this.layer[i][7] = Math.Tadd( this.layer[i][7], this.layer[i][1] );
						this.layer[i][4] = Math.Tadd( this.layer[i][4], Math.Smul(this.layer[i][6], this.learning_rate) );
						this.layer[i][2] = Math.Tadd( this.layer[i][2], Math.Smul(this.layer[i][7], this.learning_rate) );

					}break;
						
	/// neural network ############### Gradien descent without momentum ///
				default:
					for(var i=this.topology_size-1; i--;){
						if( this.type[i+1]==output_layer ){
							this.layer[i][1] = Math.Tdot( Math.act(this.layer[i][5],this.kernel[i],false), Math.cost(this.output,this.layer[i][5],this.cost) );
							this.layer[i][4] = Math.Tadd( this.layer[i][4], Math.Tmul(this.layer[i][1], Math.Smul(Math.Tsp(this.layer[i-1][5]),this.learning_rate)) );
						}
	
						else if( this.type[i]==hidden_layer ){
							this.layer[i][1] = Math.Tdot( Math.act(this.layer[i][5],this.kernel[i],false), Math.Tmul(Math.Tsp(this.layer[i+1][4]), this.layer[i+1][1]) );
							this.layer[i][4] = Math.Tadd( this.layer[i][4], Math.Tmul(this.layer[i][1], Math.Smul(Math.Tsp(this.layer[i-1][5]),this.learning_rate)) );
							this.layer[i][0] = Math.Tadd( this.layer[i][0], Math.Tmul(this.layer[i][1], Math.Smul( Math.Tsp(this.layer[i][3]),this.learning_rate)) );
						}
	
						else{
							this.layer[i][1] = Math.Tdot( Math.act(this.layer[i][5],this.kernel[i],false), Math.Tmul(Math.Tsp(this.layer[i+1][4]), this.layer[i+1][1]) );
							this.layer[i][4] = Math.Tadd( this.layer[i][4], Math.Tmul(this.layer[i][1], Math.Smul(Math.Tsp(this.input),this.learning_rate)) );
						}

						this.layer[i][2] = Math.Tadd( this.layer[i][2],Math.Smul(this.layer[i][1],this.learning_rate) );
				}break;
			}}
				
	/// neural network ############################# prvar functions ///
		clearTemp(){
			for(var i=1; i<this.topology_size ;i++){
				this.layer[i-1][3].setZero();
			}
		}
		
		randomTemp(x,y){
			for(var i=1; i<this.topology_size ;i++){
				this.layer[i-1][3].setRandom(x,y);
			}
		}
	
		showModel(){
			for(var i=0; i<this.i_layer-1; i++){
				this.layer[i][2].showTensor();
				this.layer[i][4].showTensor();
			}
		}
		
		getOutput(){ return this.layer[this.i_layer-2][5]; }
		
		showOutput(){ this.layer[this.i_layer-2][5].showTensor(); }
				
		fit( X,Y ){ this.output.setArray(Y); this.input.setArray(X); this.forwrd(); this.backwrd(); }
		
		predict( X ){ this.input.setArray(X); this.forwrd(); }
		
}	Math.RNN = _rnn;

