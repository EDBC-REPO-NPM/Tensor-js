
window.onload = ()=>{

	document.body.appendChild( Math.GPU.canvas );
	img = document.createElement('img');
	img.src = 'img_2.jpg';
	
	img.onload = ()=>{
		var image = Math.getImage(img,[640,480]);
		image = Math.HaarEdgeVer ( image,5,5 );
		Math.showImage( image );
	}
}
