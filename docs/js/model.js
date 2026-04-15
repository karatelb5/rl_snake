class NeuralNetwork {
    constructor() {
        this.layers = [];
        this.norm = null;
        this.loaded = false;
        this.inputDim = null;
    }

    async load(url) {
        try {
            const res = await fetch(url + '?t=' + Date.now(), { cache: 'no-store' });
            const data = await res.json();
            
            const keys = Object.keys(data.weights).sort((a, b) => a - b);
            keys.forEach(k => {
                this.layers.push({
                    weight: data.weights[k].weight,
                    bias: data.weights[k].bias
                });
            });

            this.norm = data.normalization;
            this.inputDim = data.architecture.input_dim;
            this.loaded = true;
        } catch (e) {
            console.error("Load error:", e);
        }
    }

    predict(input) {
        if (!this.loaded) return 0;

        if (this.inputDim && input.length !== this.inputDim) {
            return 0;
        }

        if (this.norm) {
            input = input.map((v, i) =>
                (v - this.norm.mean[i]) / Math.sqrt(this.norm.var[i] + this.norm.epsilon)
            );
        }

        let current = input;
        for (let i = 0; i < this.layers.length; i++) {
            const layer = this.layers[i];
            const next = [];
            
            for (let r = 0; r < layer.weight.length; r++) {
                let sum = layer.bias[r];
                for (let c = 0; c < current.length; c++) {
                    sum += layer.weight[r][c] * current[c];
                }
                if (i < this.layers.length - 1) sum = Math.tanh(sum);
                next.push(sum);
            }
            current = next;
        }

        let best = 0;
        for (let i = 1; i < current.length; i++) {
            if (current[i] > current[best]) best = i;
        }
        return best;
    }
}