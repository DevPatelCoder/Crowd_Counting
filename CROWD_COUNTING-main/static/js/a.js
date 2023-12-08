$(document).ready(function(){
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');
    socket.on('newnumber', function(msg) {
        console.log("Received number" + msg.number);
        $('#log').text(msg.number);
    });

});