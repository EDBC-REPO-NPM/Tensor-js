class cnn{
	public:
	/// Variables ###################################### layer state ///
		int i_layer;
		int image_size;
		int topology_size;
							
	/// Variables ################################### layer features ///
		int* knl;
		int* flt;
		int* type;
		int* depth;
					
	/// Variables #################################### learning rate ///
		int optimization;
		float learning_rate;
		float acceleration_rate;
					
	/// layer #######################################################///
		tensor* input = new tensor[3];
		tensor output, dropout;
		mlp clasifier;

		//struct for the feature extractor
		struct layer_struct{
			tensor  output;
			tensor* width; tensor* t_width;	
			tensor* bias; tensor* t_bias;	

			tensor* loss;
			tensor* conv; tensor* dIn;
			tensor* out; tensor* dOut;

		}* layer;
				
	/// neural network ################################# constructor ///
		cnn(int img_size, int n_layer=3, float n_learning_rate=0.1, int n_optimization=sgd, int n_cost_type=mean_squared):
		clasifier(3, n_learning_rate, n_optimization, n_cost_type){
			i_layer = 0;
			image_size = img_size;
			topology_size = n_layer;
			acceleration_rate = 0.9;
			optimization = n_optimization;
			learning_rate = n_learning_rate;
			
			type  = new int[topology_size];
			flt   = new int[topology_size];
			knl   = new int[topology_size-1];
			depth = new int[topology_size-1];
			layer = new layer_struct[topology_size];
		}
				
	/// neural network ########################### compile functions ///
		void add_conv_layer(int n_flt, int n_depth, int k_neuron, int n_type){
			flt[i_layer] 	= n_flt;
			knl[i_layer] 	= k_neuron;
			type[i_layer] 	= n_type;
			depth[i_layer] 	= n_depth;
			i_layer++;}

		void add_fc_layer(int n_output, int kernel){ int input_size = image_size;
			for(int i=0; i<topology_size-1 ;i++){ input_size = (input_size-flt[i]+1)/2; }
			clasifier.add_layer(input_size * input_size * depth[i_layer-1],null,input_layer);
			clasifier.input.relloc(1,input_size * input_size * depth[i_layer-1]);
			clasifier.add_layer(n_output,sigmoid,hidden_layer);
			clasifier.add_layer(n_output,kernel,output_layer);
			clasifier.output.relloc(1,n_output);
			type[i_layer] = output_layer;
			i_layer++;}
			
		void compile(){
			if(topology_size < i_layer){printf("maximo de capas superado %d cnn\n",(i_layer-topology_size));exit(1);}
			else if(topology_size > i_layer){printf("capas faltantes %d cnn\n",(topology_size-i_layer));exit(1);}
				
			for(int i=0, n_depth=1; i<topology_size-1 ;i++){
				layer[i].t_width = new tensor[depth[i]*n_depth];

				layer[i].width = new tensor[depth[i]*n_depth];
				layer[i].dOut  = new tensor[depth[i]];
				layer[i].conv  = new tensor[depth[i]];
				layer[i].loss  = new tensor[depth[i]];
				layer[i].dIn   = new tensor[depth[i]];
				layer[i].out   = new tensor[depth[i]];

				for(int j=depth[i]*n_depth; j--;){
					layer[i].t_width[j].relloc(flt[i],flt[i]); 	layer[i].t_width[j].set_random();
					layer[i].width[j].relloc(flt[i],flt[i]); 	layer[i].width[j].set_random();}

				image_size = (image_size-flt[i]+1);
				for(int j=depth[i]; j--;){
					layer[i].dIn[j].relloc(image_size+flt[i]-1,image_size+flt[i]-1);
					layer[i].dOut[j].relloc(image_size/2,image_size/2);
					layer[i].out[j].relloc(image_size/2,image_size/2);
					layer[i].conv[j].relloc(image_size,image_size);
					layer[i].loss[j].relloc(image_size,image_size);
				} image_size /= 2;

				n_depth=depth[i];}
			clasifier.compile();}
				
	/// neural network ############################# print functions ///
		void show_model(){ 
			for(int i=0; i<topology_size-1; i++){ for(int j=0; j<depth[i] ;j++){
				layer[i].width[j].show_tensor();
			}}
		}
					
		/// neural network ########################### traning functions ///
		void forwrd(){
			for(int i=0; i<topology_size; i++){

				switch(type[i]){
					case input_layer: 
						for(int j=depth[i]; j--;){ for(int k=3; --k;){
							layer[i].conv[j] += correlation( input[k], layer[i].width[ get_index(j,k,3) ] );}
					//	layer[i].conv[j] = correlation( input, layer[i].width[j] );
						layer[i].conv[j] = activation( layer[i].conv[j], knl[i], true );
						layer[i].out[j] = maxpooling( layer[i].conv[j] );
					}break;
						
					case hidden_layer: 
						for(int j=depth[i]; j--;){ for(int k=depth[i-1]; --k;){
							layer[i].conv[j] += correlation( layer[i-1].out[k], layer[i].width[ get_index(j,k,depth[i-1]) ] );}	
						layer[i].conv[j] = activation( layer[i].conv[j], knl[i], true );
						layer[i].out[j] = maxpooling( layer[i].conv[j] );
					}break;

					case output_layer:
						flatten(layer[i-1].out, depth[i-1], clasifier.input, true);
						clasifier.forwrd(); clasifier.show_output();
					break;
		}}}

		void backwrd(){
			switch( optimization ){
				case sgd: if(topology_size>2){
					for(int i=topology_size; i--;){
						switch(type[i]){
							case input_layer: for(int j=depth[i]; j--;){
								layer[i].loss[j] = activation( dmaxpooling( layer[i].conv[j], layer[i].dOut[j] ), knl[i], false );
								for(int k=3; --k;){
									layer[i].dIn[k] += convolution( padding(layer[i].loss[j], flt[i]-1), layer[i].width[ get_index(j,k,3) ]);	
									layer[i].width[ get_index(j,k,depth[i]) ] += correlation( input[k], layer[i].loss[j] ) * learning_rate;}
							//	layer[i].dIn[j]  = convolution( padding(layer[i].loss[j], flt[i]-1), layer[i].width[j]);
							//	layer[i].width[j] += correlation( input, layer[i].loss[j] ) * learning_rate;
							}break;
								
							case hidden_layer: for(int j=depth[i]; j--;){
								layer[i].loss[j] = activation( dmaxpooling( layer[i].conv[j], layer[i].dOut[j] ), knl[i], false );
								for(int k=depth[i-1]; --k;){
									layer[i-1].dOut[k] += convolution( padding(layer[i].loss[j], flt[i]-1), layer[i].width[ get_index(j,k,depth[i-1]) ]);	
									layer[i].width[ get_index(j,k,depth[i-1]) ] += correlation( layer[i-1].out[k], layer[i].loss[j] ) * learning_rate;}
							}break;
								
							case output_layer: clasifier.backwrd();
								flatten(layer[i-1].dOut, depth[i-1], clasifier.get_loss(), false );
							break;
						}
					}}

					else{ clasifier.backwrd();
						flatten(layer[0].dOut, depth[0], clasifier.get_loss(), false );
						for(int j=depth[0]; j--;){
							layer[0].loss[j]  = activation( dmaxpooling( layer[0].conv[j], layer[0].dOut[j] ), knl[0], false );
								for(int k=3; --k;){
									layer[0].dIn[k] += convolution( padding(layer[0].loss[j], flt[0]-1), layer[0].width[ get_index(j,k,3) ]);	
									layer[0].width[ get_index(j,k,depth[0]) ] += correlation( input[k], layer[0].loss[j] ) * learning_rate;}
						//	layer[0].dIn[j]   = convolution( padding(layer[0].loss[j], flt[0]-1), layer[0].width[j]);
						//	layer[0].width[j] += correlation( input, layer[0].loss[j] ) * learning_rate;
						}
					}
				break;

				case momentum: if(topology_size>2){
					for(int i=topology_size; i--;){
						switch(type[i]){
							case input_layer: for(int j=depth[i]; j--;){
								layer[i].loss[j] = activation( dmaxpooling( layer[i].conv[j], layer[i].dOut[j] ), knl[i], false );
								for(int k=3; --k;){
									layer[i].dIn[k] += convolution( padding(layer[i].loss[j], flt[i]-1), layer[i].width[ get_index(j,k,3) ]);	
									
									layer[i].t_width[j] *= acceleration_rate; 
									layer[i].t_width[get_index(j,k,3)] += correlation( input[k], layer[i].loss[j] );
									layer[i].width[get_index(j,k,3)] += layer[i].t_width[get_index(j,k,3)] * learning_rate;
								}
									
							//	layer[i].dIn[j]  = convolution( padding(layer[i].loss[j], flt[i]-1), layer[i].width[j]);
								
							//	layer[i].t_width[j] *= acceleration_rate; 
							//	layer[i].t_width[j] += correlation( input, layer[i].loss[j] );
							//	layer[i].width[j] 	+= layer[i].t_width[j] * learning_rate;
							}break;
								
							case hidden_layer: for(int j=depth[i]; j--;){
								layer[i].loss[j] = activation( dmaxpooling( layer[i].conv[j], layer[i].dOut[j] ), knl[i], false );
								for(int k=depth[i-1]; --k;){
									layer[i-1].dOut[k] += convolution( padding(layer[i].loss[j], flt[i]-1), layer[i].width[ get_index(j,k,depth[i-1]) ]);	
									
									layer[i].t_width[j] *= acceleration_rate; 
									layer[i].t_width[get_index(j,k,depth[i-1])] += correlation( layer[i-1].out[k], layer[i].loss[j] );
									layer[i].width[get_index(j,k,depth[i-1])] += layer[i].t_width[get_index(j,k,depth[i-1])] * learning_rate;
								}
							}break;
								
							case output_layer: clasifier.backwrd();
								flatten(layer[i-1].dOut, depth[i-1], clasifier.get_loss(), false );
							break;
						}
					}}

					else{ clasifier.backwrd(); 
						flatten(layer[0].dOut, depth[0], clasifier.get_loss(), false );
						for(int j=depth[0]; j--;){
							layer[0].loss[j] = activation( dmaxpooling( layer[0].conv[j], layer[0].dOut[j] ), knl[0], false );
								for(int k=3; --k;){
									layer[0].dIn[k] += convolution( padding(layer[0].loss[j], flt[0]-1), layer[0].width[ get_index(j,k,3) ]);	
									
									layer[0].t_width[j] *= acceleration_rate; 
									layer[0].t_width[get_index(j,k,3)] += correlation( input[k], layer[0].loss[j] );
									layer[0].width[get_index(j,k,3)] += layer[0].t_width[get_index(j,k,3)] * learning_rate;
								}
								
						//	layer[0].dIn[j]  = convolution( padding(layer[0].loss[j], flt[0]-1), layer[0].width[j]);
							
						//	layer[0].t_width[j] *= acceleration_rate;
						//	layer[0].t_width[j] += correlation( input, layer[0].loss[j] );
						//	layer[0].width[j] 	+= layer[0].t_width[j] * learning_rate;
						}
					}
				break;
			}
		}
			
		void fit(tensor* X, float* Y){
			input = X; clasifier.output.set_array(Y);
			forwrd(); backwrd(); }

		void predict(tensor* X){ input = X; forwrd(); }
};