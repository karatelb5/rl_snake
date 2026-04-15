class SnakeEnv {
    constructor(size = 20) {
        this.gridSize = size;
        this.maxSteps = 100 * size;
        this.reset();
    }

    reset() {
        const c = Math.floor(this.gridSize / 2);
        this.snake = [
            { x: c, y: c },
            { x: c - 1, y: c },
            { x: c - 2, y: c }
        ];
        this.direction = { x: 1, y: 0 };
        this.food = this._placeFood();
        this.score = 0;
        this.steps = 0;
        return this.getState();
    }

    _placeFood() {
        let attempts = 0;
        while (attempts < 1000) {
            const pos = {
                x: Math.floor(Math.random() * this.gridSize),
                y: Math.floor(Math.random() * this.gridSize)
            };
            if (!this.snake.some(s => s.x === pos.x && s.y === pos.y)) {
                return pos;
            }
            attempts++;
        }
        return { x: 0, y: 0 };
    }

    step(action) {
        this.steps++;
        const head = this.snake[0];

        if (action === 1) {
            this.direction = { x: -this.direction.y, y: this.direction.x };
        } else if (action === 2) {
            this.direction = { x: this.direction.y, y: -this.direction.x };
        }

        const newHead = {
            x: head.x + this.direction.x,
            y: head.y + this.direction.y
        };

        if (
            newHead.x < 0 || newHead.x >= this.gridSize ||
            newHead.y < 0 || newHead.y >= this.gridSize ||
            this.snake.some(s => s.x === newHead.x && s.y === newHead.y)
        ) {
            return { done: true, score: this.score };
        }

        this.snake.unshift(newHead);

        if (newHead.x === this.food.x && newHead.y === this.food.y) {
            this.score++;
            this.food = this._placeFood();
        } else {
            this.snake.pop();
        }

        const truncated = this.steps >= this.maxSteps;
        return { done: truncated, score: this.score };
    }

    getState() {
        const head = this.snake[0];
        const tail = this.snake[this.snake.length - 1];
        const hx = head.x, hy = head.y;
        const tx = tail.x, ty = tail.y;
        const fx = this.food.x, fy = this.food.y;
        const gs = this.gridSize;
        const d = this.direction;

        const isDanger = (dir) => {
            const nx = hx + dir.x, ny = hy + dir.y;
            return nx < 0 || nx >= gs || ny < 0 || ny >= gs ||
                   this.snake.some(s => s.x === nx && s.y === ny);
        };

        const dirStraight = d;
        const dirRight = { x: d.y, y: -d.x };
        const dirLeft = { x: -d.y, y: d.x };

        const danger = [isDanger(dirStraight)?1:0, isDanger(dirRight)?1:0, isDanger(dirLeft)?1:0];

        let dirVec = [0,0,0,0];
        if (d.y === -1) dirVec[0]=1;
        else if (d.y === 1) dirVec[1]=1;
        else if (d.x === -1) dirVec[2]=1;
        else if (d.x === 1) dirVec[3]=1;

        let foodVec = [0,0,0,0];
        if (fy < hy) foodVec[0]=1;
        else if (fy > hy) foodVec[1]=1;
        if (fx < hx) foodVec[2]=1;
        else if (fx > hx) foodVec[3]=1;

        const headX = hx / gs;
        const headY = hy / gs;
        const tailX = tx / gs;
        const tailY = ty / gs;
        const maxLen = (gs * gs) / 2;
        const snakeLen = this.snake.length / maxLen;

        return [
            ...danger,
            ...dirVec,
            ...foodVec,
            headX, headY,
            tailX, tailY,
            snakeLen
        ];
    }
}