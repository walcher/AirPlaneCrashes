function doit(){
	console.log($("#key_word").val());
	
	d3.json("data", function (e, d) {
            
    });
}
	
$( window ).load(function() {
  d3.select("#buscar2").on("click",doit);
});