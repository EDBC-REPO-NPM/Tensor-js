
const render = Math.GPU.createKernel( function(v) {
	this.color(this.thread.x / 500, this.thread.y / 500, v);
})

window.onload = ()=>{
	
	render.setOutput([500, 500]);
	render.setGraphical(true);
	
	render( 0 );
	
	document.body.appendChild( render.canvas )
	
	setInterval( ()=>{
	    render( Math.sin( Date.now() / 500) ** 2 );
	},10);

}
