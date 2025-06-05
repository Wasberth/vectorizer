// Completely based on this answer:
// http://stackoverflow.com/a/33917000
// … and the according JSfiddle:
// https://jsfiddle.net/oL2akhtz/
// 
// Enhanced for my own purposes

var dropZone = document.getElementById('dropzone');

function showDropZone() {
	dropZone.style.display = "block";
    console.log("algo")
}
function hideDropZone() {
    dropZone.style.display = "none";
}

function allowDrag(e) {
    if (true) {  // Test that the item being dragged is a valid one
        e.dataTransfer.dropEffect = 'copy';
        e.preventDefault();
    }
}

function handleDrop(e) {
    e.preventDefault();
    hideDropZone();

    alert('Drop!');
}

// 1
window.addEventListener('dragenter', function(e) {
    showDropZone();
});

// 2
window.addEventListener('dragenter', allowDrag);
window.addEventListener('dragover', allowDrag);

// 3
window.addEventListener('dragleave', function(e) {
    hideDropZone();
});

// 4
window.addEventListener('drop', handleDrop);