class _tensor {

	constructor(x=0,y=0){ 
		this.data = new Array( x*y );
		this.size = [ x,y,x*y ];
		this.data.fill(0);
	}

	new( x,y ){ 
		if( this.data != null ) delete this.data;
		this.data = new Array( x*y );
		this.size = [ x,y,x*y ];
		this.data.fill(0);
	}

	XLoop( _callback ){ for( var x=0; x<this.size[0]; x++ ){ const a=_callback(x); if(a==0)break; else if(a==1)continue; }}
	YLoop( _callback ){ for( var y=0; y<this.size[1]; y++ ){ const a=_callback(y); if(a==0)break; else if(a==1)continue; }}
	ZLoop( _callback ){ for( var z=0; z<this.size[2]; z++ ){ const a=_callback(z); if(a==0)break; else if(a==1)continue; }}
	_XLoop( _callback ){ for( var x=this.size[0]; x--; ){ const a=_callback(x); if(a==0)break; else if(a==1)continue; }}
	_YLoop( _callback ){ for( var y=this.size[1]; y--; ){ const a=_callback(y); if(a==0)break; else if(a==1)continue; }}
	_ZLoop( _callback ){ for( var z=this.size[2]; z--; ){ const a=_callback(z); if(a==0)break; else if(a==1)continue; }}

	random( x=1,b=0 ){ this.data = this.data.map( v=>Math.random()*x+b ); }
	array(a){ this.data = this.data.map( (x,i)=>a[i] ); }
	order(){ this.data = this.data.map( (x,i)=>i ); }
	value(v){ this.data.fill(v); }
	zero(){ this.data.fill(0); }
	one(){ this.data.fill(1); }

	getValue(x,y){ return this.data[ this.getIndex(x,y) ]; }
	setValue(v,x,y){ this.data[ this.getIndex(x,y) ] = v; }
	getIndex(x,y){ return x+(this.size[0]*y); }

	shape(x,y){ 
		this.size = [ x,y,x*y ]; 
		this.data = this.data.slice(0,x*y); 
		if( this.data.length < x*y ){
			const diff = Math.abs( this.data.length - x*y );
			const A = new Array( diff ); A.fill(0)
			this.data.push( ...A );
		}
	}
	
	show(_s='\t',_float=0){
		var log = `Shape:[${this.size[0]}, ${this.size[1]}] \n`;
		this.YLoop( (y)=>{ this.XLoop( (x)=>{
			if( this.getValue(x,y)>=0 ) log += `${_s}`;
			log += `${ this.getValue(x,y)?.toFixed(_float) }`;
		}); log += '\n'; }); console.log( log );
	}

	transpose(){
		const _tns = new Math.tensor( this.size[1],this.size[0] );
		this.YLoop( (y)=>{ this.XLoop( (x)=>{
			_tns.setValue(this.getValue(y,x),x,y);
		}); });	return _tns;
	}

	flipXY(){
		const _tns = new Math.tensor( this.size[0],this.size[1] );
		this.YLoop( (y)=>{ this.XLoop( (x)=>{
			const _x = this.size[0] - 1 - x;
			const _y = this.size[1] - 1 - y;
			_tns.setValue(this.getValue(_x,_y),x,y);
		}); });	return _tns;
	}

	flipX(){
		const _tns = new Math.tensor( this.size[0],this.size[1] );
		this.YLoop( (y)=>{ this.XLoop( (x)=>{
			const _x = this.size[0] - 1 - x;
			_tns.setValue(this.getValue(_x,y),x,y);
		}); });	return _tns;
	}

	flipY(){
		const _tns = new Math.tensor( this.size[0],this.size[1] );
		this.YLoop( (y)=>{ this.XLoop( (x)=>{
			const _y = this.size[1] - 1 - y;
			_tns.setValue(this.getValue(x,_y),x,y);
		}); });	return _tns;
	}

	clone(){
		const _tns = new Math.tensor( this.size[0],this.size[1] );
		_tns.array( this.data );
		return _tns;
	}

	pushPadding( _i=1 ){ for( var i=_i; i--; ){
		this.unshiftRow(); this.unshiftCol();
		this.pushRow(); this.pushCol();
	}	return this; }

	popPadding( _i=1 ){ for( var i=_i; i--; ){
		this.shiftRow(); this.shiftCol();
		this.popRow(); this.popCol();
	}	return this; }

	unshiftRow( _i=1 ){ for( var i=_i; i--; ){
		this._YLoop( (y)=>{ this._XLoop( (x)=>{
			const p = this.getIndex(x,y);
			if( x==0 ) this.data.splice( p,0,0 );
		}); }); this.size[0] += _i; this.size[2]  = this.size[0] * this.size[1];
	}	return this; }

	shiftRow( _i=1 ){ for( var i=_i; i--; ){
		this._YLoop( (y)=>{ this._XLoop( (x)=>{
			const p = this.getIndex(x,y);
			if( x==0 ) this.data.splice( p,1 );
		}); }); this.size[0] -= _i; this.size[2]  = this.size[0] * this.size[1];
	}	return this; }

	pushRow( _i=1 ){ for( var i=_i; i--; ){
		this._YLoop( (y)=>{ this._XLoop( (x)=>{
			const p = this.getIndex(x,y);
			if( x==this.size[0]-1 ) this.data.splice( p+1,0,0 );
		}); }); this.size[0] += _i; this.size[2]  = this.size[0] * this.size[1];
	}	return this; }

	popRow( _i=1 ){ for( var i=_i; i--; ){
		this._YLoop( (y)=>{ this._XLoop( (x)=>{
			const p = this.getIndex(x,y);
			if( x==this.size[0]-1 ) this.data.splice( p,1 );
		}); }); this.size[0] -= _i; this.size[2]  = this.size[0] * this.size[1];
	}	return this; }

	unshiftCol( _i=1 ){ for( var i=_i; i--; ){
		const A = new Array( this.size[0] ); A.fill(0);
		this.data.unshift( ...A ); this.size[1] += _i;
		this.size[2]  = this.size[0] * this.size[1];
	}	return this; }

	shiftCol( _i=1 ){ for( var i=_i; i--; ){
		this.data.splice( 0,this.size[0] );
		this.size[1] -= _i; this.size[2]  = this.size[0] * this.size[1];
	}	return this; }

	pushCol( _i=1 ){ for( var i=_i; i--; ){
		const A = new Array( this.size[0] ); A.fill(0);
		this.data.push( ...A ); this.size[1] += _i;
		this.size[2]  = this.size[0] * this.size[1];
	}	return this; }

	popCol( _i=1 ){ for( var i=_i; i--; ){
		this.data.splice( this.size[2]-this.size[0],this.size[2] );
		this.size[1] -= _i; this.size[2]  = this.size[0] * this.size[1];
	}	return this; }


	//TODO: Math
	min(){ return Math.min( ...this.data ) }
	max(){ return Math.max( ...this.data ) }

	sin(){ this.data = this.data.map( x=>Math.sin(x) ); return this; }
	
	asin(){ this.data = this.data.map( x=>Math.asin(x)); return this; }
	sinh(){ this.data = this.data.map( x=>Math.sinh(x)); return this; }
	asinh(){ this.data = this.data.map( x=>Math.asinh(x)); return this;}

	cos(){ this.data = this.data.map( x=>Math.cos(x)); return this;}
	acos(){ this.data = this.data.map( x=>Math.acos(x)); return this;}
	cosh(){ this.data = this.data.map( x=>Math.cosh(x)); return this;}
	acosh(){ this.data = this.data.map( x=>Math.acosh(x)); return this;}

	tan(){ this.data = this.data.map( x=>Math.tan(x)); return this;}
	atan(){ this.data = this.data.map( x=>Math.atan(x)); return this;}
	tanh(){ this.data = this.data.map( x=>Math.tanh(x)); return this;}
	atanh(){ this.data = this.data.map( x=>Math.atanh(x)); return this;}

	log(){ this.data = this.data.map( x=>Math.log(x)); return this;}
	log2(){ this.data = this.data.map( x=>Math.log2(x)); return this;}
	log10(){ this.data = this.data.map( x=>Math.log10(x)); return this;}

	abs(){ this.data = this.data.map( x=>Math.abs(x)); return this;}
	ceil(){ this.data = this.data.map( x=>Math.ceil(x)); return this;}
	sign(){ this.data = this.data.map( x=>Math.sign(x)); return this;}
	sqrt(){ this.data = this.data.map( x=>Math.sqrt(x)); return this;}
	round(){ this.data = this.data.map( x=>Math.round(x)); return this;}
	floor(){ this.data = this.data.map( x=>Math.floor(x)); return this;}
	trunc(){ this.data = this.data.map( x=>Math.trunc(x)); return this;}

	pow(e){ this.data = this.data.map( x=>Math.pow(x,e)); return this;}


	//TODO: Activation

} 	

Math.tensor = _tensor;
module.exports = _tensor;