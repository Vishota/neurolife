export default class Simulation {
    device: GPUDevice
    pipeline: GPUComputePipeline
    bindGroup: GPUBindGroup

    weightsBuffer: GPUBuffer
    inputsBuffer: GPUBuffer
    outputsBuffer: GPUBuffer
    cellsBuffer: GPUBuffer
    foodBuffer: GPUBuffer
    globalDataBuffer: GPUBuffer
    cameraBuffer: GPUBuffer

    renderPipeline: GPURenderPipeline
    renderBindGroup: GPUBindGroup
    format: GPUTextureFormat

    scale: number = 1
    camY: number = 0
    camX: number = 0
    logTimer = 0
    currentInstanceCount: number = 0

    canvas: HTMLCanvasElement
    context: GPUCanvasContext

    cellCount: number

    async initialize() {
        const maxCells = this.cellCount
        const maxFood = 100000//todo const management...

        // ---------- GLOBAL ----------
        const globalData = new Uint32Array(3)
        globalData[0] = this.cellCount // стартовое количество клеток
        globalData[1] = 0 // стартовое количество еды
        globalData[2] = 0

        this.device.queue.writeBuffer(
            this.globalDataBuffer,
            0,
            globalData
        )

        // ---------- CELLS ----------
        const cellStride = 5 // x,y,energy,mem0-1
        const cells = new Float32Array(maxCells * cellStride)

        for (let i = 0; i < globalData[0]; i++) {
            const base = i * cellStride

            cells[base + 0] = (Math.random() - 0.5) * 500
            cells[base + 1] = (Math.random() - 0.5) * 500
            cells[base + 2] = 1000

            cells[base + 3] = 0
            cells[base + 4] = 0
        }

        this.device.queue.writeBuffer(
            this.cellsBuffer,
            0,
            cells
        )

        // ---------- FOOD ----------
        const foodStride = 2
        const food = new Float32Array(maxFood * foodStride)

        for (let i = 0; i < globalData[1]; i++) {
            const base = i * foodStride
            food[base + 0] = (Math.random() - 0.5) * 50
            food[base + 1] = (Math.random() - 0.5) * 50
        }

        this.device.queue.writeBuffer(
            this.foodBuffer,
            0,
            food
        )

        // ---------- WEIGHTS ----------
        const weightsSize = this.weightsBuffer.size / 4
        const weights = new Float32Array(weightsSize)

        for (let i = 0; i < weights.length; i++) {
            weights[i] = (Math.random() * 2 - 1) * 0.5
        }

        this.device.queue.writeBuffer(
            this.weightsBuffer,
            0,
            weights
        )

        // камера
        this.device.queue.writeBuffer(
            this.cameraBuffer,
            0,
            new Float32Array([0, 0, 1, 0])
        )
    }

    async logState() {
        const readBuffer = this.device.createBuffer({
            size: 12,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        })

        const encoder = this.device.createCommandEncoder()
        encoder.copyBufferToBuffer(
            this.globalDataBuffer,
            0,
            readBuffer,
            0,
            12
        )

        this.device.queue.submit([encoder.finish()])

        await readBuffer.mapAsync(GPUMapMode.READ)

        const data = new Uint32Array(readBuffer.getMappedRange())
        const cells = data[0]
        const food = data[1]

        console.log('Cells:', cells, 'Food:', food)

        readBuffer.unmap()
    }

    constructor(
        device: GPUDevice,
        layers: number[],
        cellCount: number,
        canvas: HTMLCanvasElement
    ) {
        this.device = device
        this.cellCount = cellCount
        this.canvas = canvas
        this.context = canvas.getContext('webgpu')!

        const cellWeightsVolume = layers
            .slice(0, -1)
            .reduce((sum, layer, i) => sum + layer * layers[i + 1], 0)

        const maxLayerSize = Math.max(...layers)

        const maxFoodCount = 100000

        const cellSpeed = 1

        const shader = `
fn rand(seed: ptr<function, u32>) -> f32 {
    (*seed) = (*seed) * 1664525u + 1013904223u;
    return f32((*seed) & 0x00FFFFFFu) / f32(0x01000000u);
}




const vision_distance = 100;

const world_size: f32 = 1000.0;
const half_world: f32 = world_size * 0.5;
const eat_distance: f32 = 4.0;
const energy_from_food: f32 = 140.0;
const food_spawn_per_tick: u32 = 9u;

fn wrap_delta(d: f32) -> f32 {
    var delta = d;

    if (delta > half_world) {
        delta -= world_size;
    }
    if (delta < -half_world) {
        delta += world_size;
    }

    return delta;
}

fn torus_distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = wrap_delta(x1 - x2);
    let dy = wrap_delta(y1 - y2);
    return sqrt(dx*dx + dy*dy);
}

const layer_volumes: array<u32, ${layers.length}> =
    array<u32, ${layers.length}>(${layers.join(',')});

struct global_data {
    cells_count: u32,
    food_count: u32,
    tick: u32,
};

// global data
@group(0) @binding(0) var<storage, read_write> global: global_data;

// neuro
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> inputs: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;


// cells
struct cell_params {
    x: f32,
    y: f32,
	energy: f32,
	mem0: f32,
	mem1: f32
};
@group(0) @binding(4) var<storage, read_write> cells: array<cell_params>;

struct food_params {
    x: f32,
    y: f32
};


// food
@group(0) @binding(5) var<storage, read_write> food_data: array<food_params>;

struct Camera {
    pos: vec2<f32>,
    scale: f32,
    _pad: f32,
};


fn neuron_id(cell: u32, layer: u32, neuron: u32, weight: u32) -> u32 {
    var prev_layers_volume: u32 = 0;
    for (var i: u32 = 0; i < layer; i = i + 1) {
        prev_layers_volume += layer_volumes[i] * layer_volumes[i + 1];
    }

    let next_layer_volume: u32 = layer_volumes[layer + 1];

    return cell * ${cellWeightsVolume}
        + prev_layers_volume
        + neuron * next_layer_volume
        + weight;
}

fn in_id(cell: u32, i: u32) -> u32 {
    return cell * ${layers[0]} + i;
}

fn out_id(cell: u32, o: u32) -> u32 {
    return cell * ${layers[layers.length - 1]} + o;
}

fn prepare_inputs_cells(id: vec3<u32>) {
	let cur_cell = cells[id.x];
    let cur_gridX = ceil(cur_cell.x / vision_distance);
    let cur_gridY = ceil(cur_cell.y / vision_distance);
    

    var closest0:f32 = 1000000;
    var closest1:f32 = 1000000;

    for(var i:u32 = 0; i < global.cells_count; i++) {
        if(i == id.x) {
            continue;
        }

        let n_physics = cells[i];
        let n_gridX = ceil(n_physics.x / vision_distance);
        let n_gridY = ceil(n_physics.y / vision_distance);

        let is_potentially_visible = 
            cur_gridX > n_gridX - 1 &&
            cur_gridX < n_gridX + 1 &&
            cur_gridY > n_gridY - 1 &&
            cur_gridY < n_gridY + 1;

        if(!is_potentially_visible) {
            continue;
        }

        let distance = torus_distance(
            cur_cell.x,
            cur_cell.y,
            n_physics.x,
            n_physics.y
        );

        if(distance > vision_distance) {
            continue;
        }

        if(distance < closest0) {
            closest1 = closest0;
            closest0 = distance;
            continue;
        }
    };

    set_input(closest0, id.x, 0);
}

fn prepare_inputs_food(id: vec3<u32>) {
	let cur_cell = cells[id.x];
    let cur_gridX = ceil(cur_cell.x / vision_distance);
    let cur_gridY = ceil(cur_cell.y / vision_distance);
    
    var closest0:f32 = 1000000;
    var closest1:f32 = 1000000;

    for(var i:u32 = 0; i < global.food_count; i++) {
        let food_physics = food_data[i];
        let food_gridX = ceil(food_physics.x / vision_distance);
        let food_gridY = ceil(food_physics.y / vision_distance);

        let is_potentially_visible = 
            cur_gridX > food_gridX - 1 &&
            cur_gridX < food_gridX + 1 &&
            cur_gridY > food_gridY - 1 &&
            cur_gridY < food_gridY + 1;

        if(!is_potentially_visible) {
            continue;
        }

        let distance = torus_distance(
            cur_cell.x,
            cur_cell.y,
            food_physics.x,
            food_physics.y
        );

        if(distance < closest0) {
            closest1 = closest0;
            closest0 = distance;
            continue;
        }
    };

    set_input(closest0, id.x, 1);
}

fn prepare_inputs_energy(id: vec3<u32>) {
	let cur_cell = cells[id.x];
	
    set_input(cur_cell.energy, id.x, 2);
}

fn prepare_inputs_memory(id: vec3<u32>) {
	let cur_cell = cells[id.x];
	
    set_input(cur_cell.mem0, id.x, 3);
    set_input(cur_cell.mem1, id.x, 4);
}

fn prepare_inputs(id: vec3<u32>) {
	prepare_inputs_cells(id);
	prepare_inputs_food(id);
	prepare_inputs_energy(id);
	prepare_inputs_memory(id);
}

// ==========INPUTS==========
// 0	= dist to cell0
// 1	= dist to food0
// 2	= energy
// 3	= mem0
// 4	= mem1
// ==========================


fn set_input(raw_value: f32, cell: u32, number: u32) {
    inputs[in_id(cell, number)] = tanh(raw_value);
}
fn get_output(cell: u32, number: u32) -> f32 {
    return outputs[out_id(cell, number)];
}


fn calc_neuro(id: vec3<u32>) {
    let cell: u32 = u32(id.x);

    var current: array<f32, ${maxLayerSize}>;
    var next: array<f32, ${maxLayerSize}>;

    for (var i: u32 = 0; i < layer_volumes[0]; i = i + 1) {
        current[i] = inputs[in_id(cell, i)];
    }

    for (var layer: u32 = 0; layer < ${layers.length - 1}; layer = layer + 1) {

        let inSize: u32 = layer_volumes[layer];
        let outSize: u32 = layer_volumes[layer + 1];

        for (var o: u32 = 0; o < outSize; o = o + 1) {

            var sum: f32 = 0.0;

            for (var i: u32 = 0; i < inSize; i = i + 1) {
                let wIndex = neuron_id(cell, layer, o, i);
                sum = sum + current[i] * weights[wIndex];
            }

            next[o] = tanh(sum);
        }

        for (var k: u32 = 0; k < outSize; k = k + 1) {
            current[k] = next[k];
        }
    }

    let final_size: u32 = layer_volumes[${layers.length - 1}];

    for (var o: u32 = 0; o < final_size; o = o + 1) {
        outputs[out_id(cell, o)] = current[o];
    }
}



fn mutate_weights(parent: u32, child: u32) {
    var seed: u32 = parent * 9781u + child * 6271u;

    let mutation_probability: f32 = .5;
    let mutation_strength: f32 = .1;

    for (var w: u32 = 0u; w < ${cellWeightsVolume}u; w = w + 1u) {
        let start = parent * ${cellWeightsVolume}u + w;
        let end   = child  * ${cellWeightsVolume}u + w;

        weights[end] = weights[start];

        if (rand(&seed) < mutation_probability) {
            let delta = (rand(&seed) * 2.0 - 1.0) * mutation_strength;
            weights[end] = weights[end] + delta;
        }
    }
}






fn divide(id: vec3<u32>) {
    let parent = id.x;

    if (global.cells_count >= 100000) { // todo const cells count limit
        return;
    }

    if (cells[parent].energy < 0.5) {
        return;
    }

    let child = global.cells_count;

    cells[child] = cells[parent];

    //cells[child].x += 2.0;
    //cells[child].y += 2.0;

    cells[parent].energy -= 3; // todo divide cost

    cells[child].energy = cells[parent].energy * 0.5;
    cells[parent].energy *= 0.5;

	cells[child].mem0 = 0f;
	cells[child].mem1 = 0f;

    mutate_weights(parent, child);

    global.cells_count = global.cells_count + 1u;
}
fn share(id: vec3<u32>) {
	// todo later
}
fn kill(id: vec3<u32>) {
    let i = id.x;

    if (global.cells_count == 0u) {
        return;
    }

    let last = global.cells_count - 1u;

    if (i != last) {
        cells[i] = cells[last];

        for (var w: u32 = 0u; w < ${cellWeightsVolume}u; w = w + 1u) {
            let start = last * ${cellWeightsVolume}u + w;
            let end = i * ${cellWeightsVolume}u + w;
            weights[end] = weights[start];
        }
    }

    global.cells_count = global.cells_count - 1u;
}

fn use_outputs(id: vec3<u32>) {
    var max_linear_speed = ${cellSpeed}f;
	let i = id.x;

	var vx = get_output(i, 0);
	var vy = get_output(i, 1);

	let vlen = sqrt(vx * vx + vy * vy);

	if (vlen > 1.0) {
	    vx = vx / vlen;
	    vy = vy / vlen;
	}

	cells[i].x += vx * max_linear_speed;
    cells[i].y += vy * max_linear_speed;
     
    if (cells[i].x > half_world) {
        cells[i].x -= world_size;
    }
    if (cells[i].x < -half_world) {
        cells[i].x += world_size;
    }
    if (cells[i].y > half_world) {
        cells[i].y -= world_size;
    }
    if (cells[i].y < -half_world) {
        cells[i].y += world_size;
    }

	let energy_loss_const = 1f;
	let energy_loss_movement = 2f;

	cells[i].energy -= energy_loss_const + energy_loss_movement * min(1, vlen);

	if(cells[i].energy < 0) {
		kill(id);
		return;
	}

	let act_divide = get_output(i, 2);
	let act_exist = get_output(i, 3);

	if(act_divide > act_exist) {
		divide(id);
	}

	var dmem0 = get_output(i, 4);
	var dmem1 = get_output(i, 5);

	cells[i].mem0 = tanh(cells[i].mem0 * 0.97 + dmem0 * 0.05); // todo set coefficents
	cells[i].mem1 = tanh(cells[i].mem1 * 0.97 + dmem1 * 0.05);



    let cur_cell = cells[i];

    let cur_gridX = ceil(cur_cell.x / vision_distance);
    let cur_gridY = ceil(cur_cell.y / vision_distance);

    for (var f: u32 = 0u; f < global.food_count; f = f + 1u) {

        let food_physics = food_data[f];

        let food_gridX = ceil(food_physics.x / vision_distance);
        let food_gridY = ceil(food_physics.y / vision_distance);

        let is_potentially_close =
            cur_gridX > food_gridX - 1.0 &&
            cur_gridX < food_gridX + 1.0 &&
            cur_gridY > food_gridY - 1.0 &&
            cur_gridY < food_gridY + 1.0;

        if (!is_potentially_close) {
            continue;
        }

        let dist = torus_distance(
            cur_cell.x,
            cur_cell.y,
            food_physics.x,
            food_physics.y
        );

        if (dist < eat_distance) {

            // add energy (more energy we have - less energy we gain)
            cells[i].energy += energy_from_food * (1 - tanh(cells[i].energy / 10000));

            // remove food (swap with last)
            let last = global.food_count - 1u;

            if (f != last) {
                food_data[f] = food_data[last];
            }

            global.food_count = global.food_count - 1u;

            break;
        }
    }
}

// ==========OUTPUTS=========
// 0	= speed X
// 1	= speed Y
//			<<<ACTIONS>>>
// 2	= divide
// 3	= exist (do nothing)
//			<<<MEMORY>>>
// 4	= mem0
// 5	= mem1
// ==========================


fn spawn_food() {
    var seed: u32 = global.tick * 747796405u + 2891336453u;

    for (var i: u32 = 0u; i < food_spawn_per_tick; i = i + 1u) {

        if (global.food_count >= ${maxFoodCount}u) {
            return;
        }

        let index = global.food_count;

        let rx = rand(&seed);
        let ry = rand(&seed);

        food_data[index].x = rx * world_size * .5 - half_world * .5;
        food_data[index].y = ry * world_size * .5 - half_world * .5;

        global.food_count = global.food_count + 1u;
    }
}


@compute @workgroup_size(1)
fn tick(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x == 0u) {
        spawn_food();
        global.tick = global.tick + 1u;
    }

    if (id.x >= global.cells_count) {
        return;
    }

    prepare_inputs(id);
    calc_neuro(id);
    use_outputs(id);
}
`

        const module = device.createShaderModule({ code: shader })

        this.pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module,
                entryPoint: 'tick'
            }
        })

        const inputsSize =
            cellCount * layers[0] * 4

        const outputsSize =
            cellCount * layers[layers.length - 1] * 4

        this.weightsBuffer = device.createBuffer({
            size: 134217728, //todo change
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_DST
        })

        this.inputsBuffer = device.createBuffer({
            size: inputsSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_DST
        })

        this.outputsBuffer = device.createBuffer({
            size: outputsSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })

        this.cellsBuffer = device.createBuffer({
            size: 134217728, //todo change
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })

        this.foodBuffer = device.createBuffer({
            size: 134217728, //todo change
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })

        this.globalDataBuffer = device.createBuffer({
            size: 12,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        })

        this.cameraBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_DST
        })

        this.bindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.globalDataBuffer } },
                { binding: 1, resource: { buffer: this.weightsBuffer } },
                { binding: 2, resource: { buffer: this.inputsBuffer } },
                { binding: 3, resource: { buffer: this.outputsBuffer } },
                { binding: 4, resource: { buffer: this.cellsBuffer } },
                { binding: 5, resource: { buffer: this.foodBuffer } },
            ]
        })

        this.format = navigator.gpu.getPreferredCanvasFormat()

        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'opaque'
        })

        const renderShader = `
struct cell_params {
    x: f32,
    y: f32,
    energy: f32,
    mem0: f32,
    mem1: f32
};

struct food_params {
    x: f32,
    y: f32
};

struct Camera {
    pos: vec2<f32>,
    scale: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> cells: array<cell_params>;
@group(0) @binding(1) var<storage, read> food_data: array<food_params>;
@group(0) @binding(2) var<storage, read> global_data: vec2<u32>;
@group(0) @binding(3) var<uniform> camera: Camera;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>
};

@vertex
fn vs_main(
    @builtin(instance_index) instance: u32,
    @builtin(vertex_index) vertexIndex: u32
) -> VSOut {

    var positions = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0)
    );

    let quad = positions[vertexIndex];

    var out: VSOut;

    if (instance < global_data.x) {
        let cell = cells[instance];

        let size = 1.0;

        let worldPos = vec2(cell.x, cell.y) + quad * size;

        let view = (worldPos - camera.pos) * camera.scale;

        let ndc = vec2(
            view.x / 500.0,
            -view.y / 500.0
        );

        out.position = vec4(ndc, 0.0, 1.0);

        let e = clamp(cell.energy * 0.001, 0.0, 3.0);
        out.color = vec3(e, e - 1, e - 2);

        return out;
    }

    let foodIndex = instance - global_data.x;

    let food = food_data[foodIndex];

    let size = 1.0;

    let worldPos = vec2(food.x, food.y) + quad * size;

    let view = (worldPos - camera.pos) * camera.scale;

    let ndc = vec2(
        view.x / 500.0,
        -view.y / 500.0
    );

    out.position = vec4(ndc, 0.0, 1.0);
    out.color = vec3(0.0, 1.0, 0.0);

    return out;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
    return vec4(input.color, 1.0);
}
`

        const renderModule = this.device.createShaderModule({ code: renderShader })

        this.renderPipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main'
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{ format: this.format }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        })

        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.cellsBuffer } },
                { binding: 1, resource: { buffer: this.foodBuffer } },
                { binding: 2, resource: { buffer: this.globalDataBuffer } },
                { binding: 3, resource: { buffer: this.cameraBuffer } }
            ]
        })


        window.addEventListener('keydown', e => {
            const speed = 10 / this.scale

            if (e.key === 'w') this.camY -= speed
            if (e.key === 's') this.camY += speed
            if (e.key === 'a') this.camX -= speed
            if (e.key === 'd') this.camX += speed

            if (e.key === 'q') this.scale *= 1.1
            if (e.key === 'e') this.scale *= 0.9

            this.device.queue.writeBuffer(
                this.cameraBuffer,
                0,
                new Float32Array([this.camX, this.camY, this.scale, 0])
            )
        })

        this.initialize()
    }

    tick() {
        const encoder = this.device.createCommandEncoder()
        const pass = encoder.beginComputePass()

        pass.setPipeline(this.pipeline)
        pass.setBindGroup(0, this.bindGroup)

        pass.dispatchWorkgroups(Math.max(this.cellCount, 1))

        pass.end()

        this.device.queue.submit([encoder.finish()])
    }

    async updateInstanceCount() {
        const readBuffer = this.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        })

        const encoder = this.device.createCommandEncoder()
        encoder.copyBufferToBuffer(
            this.globalDataBuffer,
            0,
            readBuffer,
            0,
            8
        )

        this.device.queue.submit([encoder.finish()])

        await readBuffer.mapAsync(GPUMapMode.READ)

        const data = new Uint32Array(readBuffer.getMappedRange())
        const cells = data[0]
        const food = data[1]

        this.currentInstanceCount = cells + food

        readBuffer.unmap()
    }

    render() {
        const textureView = this.context.getCurrentTexture().createView()

        const encoder = this.device.createCommandEncoder()

        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: textureView,
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: 'clear',
                    storeOp: 'store'
                }
            ]
        })

        pass.setPipeline(this.renderPipeline)
        pass.setBindGroup(0, this.renderBindGroup)

        pass.draw(6, this.currentInstanceCount)

        pass.end()

        this.device.queue.submit([encoder.finish()])
    }

    async loop() {
        this.tick()
        await this.updateInstanceCount()
        this.render()

        this.logTimer++
        if (this.logTimer > 60) {
            this.logTimer = 0
            this.logState()
        }

        requestAnimationFrame(() => this.loop())
    }
}