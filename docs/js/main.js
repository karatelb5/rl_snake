const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

const env = new SnakeEnv(20);
const model = new NeuralNetwork();

let interval = null;
let isPaused = false;
let currentSpeed = 150;

function draw() {
    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = '#141414';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 20; i++) {
        ctx.beginPath();
        ctx.moveTo(i * 20, 0);
        ctx.lineTo(i * 20, 400);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i * 20);
        ctx.lineTo(400, i * 20);
        ctx.stroke();
    }

    const fx = env.food.x * 20 + 10;
    const fy = env.food.y * 20 + 10;
    
    const glow = ctx.createRadialGradient(fx, fy, 2, fx, fy, 15);
    glow.addColorStop(0, 'rgba(239, 68, 68, 0.6)');
    glow.addColorStop(1, 'rgba(239, 68, 68, 0)');
    ctx.fillStyle = glow;
    ctx.fillRect(env.food.x * 20 - 5, env.food.y * 20 - 5, 30, 30);

    ctx.fillStyle = '#ef4444';
    ctx.fillRect(env.food.x * 20 + 4, env.food.y * 20 + 4, 12, 12);

    env.snake.forEach((s, i) => {
        const brightness = 1 - (i / Math.max(env.snake.length, 10)) * 0.6;
        ctx.fillStyle = `rgba(220, 38, 38, ${brightness})`;
        ctx.fillRect(s.x * 20 + 1, s.y * 20 + 1, 18, 18);
        ctx.strokeStyle = '#7f1d1d';
        ctx.lineWidth = 1;
        ctx.strokeRect(s.x * 20 + 1, s.y * 20 + 1, 18, 18);
    });

    if (env.snake.length > 0) {
        const head = env.snake[0];
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(head.x * 20 + 6, head.y * 20 + 6, 3, 3);
        ctx.fillRect(head.x * 20 + 11, head.y * 20 + 6, 3, 3);
    }
}

function updateUI() {
    const scoreEl = document.getElementById('scoreDisplay');
    const stepsEl = document.getElementById('stepsDisplay');
    
    if (scoreEl) scoreEl.innerText = env.score;
    if (stepsEl) stepsEl.innerText = env.steps;
}

function startGame(speed = currentSpeed) {
    if (interval) clearInterval(interval);
    env.reset();
    draw();
    updateUI();
    interval = setInterval(gameLoop, speed);
}

function gameLoop() {
    if (isPaused) return;
    runStep();
}

function runStep() {
    if (!model.loaded) return;
    
    const state = env.getState();
    const action = model.predict(state);
    const res = env.step(action);
    
    draw();
    updateUI();
    
    if (res.done) {
        clearInterval(interval);
        setTimeout(() => startGame(currentSpeed), 1000);
    }
}

function togglePause() {
    isPaused = !isPaused;
}

function setSpeed(speed) {
    currentSpeed = speed;
    if (interval) {
        clearInterval(interval);
        if (!isPaused) {
            interval = setInterval(gameLoop, speed);
        }
    }
}

window.onload = async () => {
    try {
        await model.load('models/weights.json');
        
        document.getElementById('btnStart').onclick = () => startGame(currentSpeed);
        document.getElementById('btnReset').onclick = () => {
            clearInterval(interval);
            env.reset();
            draw();
            updateUI();
        };
        document.getElementById('btnPause').onclick = togglePause;
        
        const speedRange = document.getElementById('speedRange');
        const speedValue = document.getElementById('speedValue');
        if (speedRange && speedValue) {
            speedRange.oninput = (e) => {
                const ms = 500 - (e.target.value * 4.5);
                speedValue.textContent = Math.round(ms);
                setSpeed(ms);
            };
        }
        
        draw();
        updateUI();
        
    } catch (err) {
        console.error("Error:", err);
        alert("Model load failed");
    }
};