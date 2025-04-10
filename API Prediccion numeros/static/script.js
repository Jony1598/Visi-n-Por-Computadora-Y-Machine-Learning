const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let painting = false;

canvas.addEventListener('mousedown', () => painting = true);
canvas.addEventListener('mouseup', () => painting = false);
canvas.addEventListener('mouseleave', () => painting = false);
canvas.addEventListener('mousemove', draw);

function draw(e) {
    if (!painting) return;
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById("resultado").textContent = "Resultado: ";
}

function sendImage() {
    const dataURL = canvas.toDataURL("image/png");
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imagen: dataURL })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("resultado").textContent = `Resultado: ${data.prediction}`;
    })
    .catch(err => console.error(err));
}

