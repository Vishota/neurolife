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

	scale: number = 1
	camY: number = 0
	camX: number = 0

	canvas: HTMLCanvasElement
	context: GPUCanvasContext

	cellCount: number

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

		const maxCellsCount = 1000
		const maxFoodCount = 1000

		const cellSpeed = 5

		const shader = `
fn rand(seed: ptr<function, u32>) -> f32 {
    (*seed) = (*seed) * 1664525u + 1013904223u;
    return f32((*seed) & 0x00FFFFFFu) / f32(0x01000000u);
}




const vision_distance = 30;
const layer_volumes: array<u32, ${layers.length}> =
    array<u32, ${layers.length}>(${layers.join(',')});

struct global_data {
    cells_count: u32,
    food_count: u32,
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
	mem1: f32,
	mem2: f32,
	mem3: f32,
	mem4: f32
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

@group(0) @binding(6)
var<uniform> camera: Camera;


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

    var distances: array<f32, ${maxCellsCount}>;
    var distances_count = 0u;
    
    for(var i:u32 = 0; i < global.cells_count; i++) {
        distances[i] = -1;

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

        let dx = abs(cur_cell.x - n_physics.x);
        let dy = abs(cur_cell.y - n_physics.y);

        let distance = sqrt(dx*dx+dy*dy);

        if(distance > vision_distance) {
            continue;
        }

        distances[distances_count] = distance;
        distances_count++;
    };

    var closest0:f32 = -1;
    var closest1:f32 = -1;

    for (var i:u32 = 0; i < distances_count; i++) {
        if(distances[i] < closest0) {
            closest1 = closest0;
            closest0 = distances[i];
            continue;
        }
    }

    set_input(closest0, id.x, 0);
    set_input(closest1, id.x, 1);
}

fn prepare_inputs_food(id: vec3<u32>) {
	let cur_cell = cells[id.x];
    let cur_gridX = ceil(cur_cell.x / vision_distance);
    let cur_gridY = ceil(cur_cell.y / vision_distance);

    var distances: array<f32, ${maxFoodCount}>;
    var distances_count = 0u;
    
    for(var i:u32 = 0; i < global.food_count; i++) {
        distances[i] = -1;

        if(i == id.x) {
            continue;
        }

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

        let dx = abs(cur_cell.x - food_physics.x);
        let dy = abs(cur_cell.y - food_physics.y);

        let distance = sqrt(dx*dx+dy*dy);

        if(distance > vision_distance) {
            continue;
        }

        distances[distances_count] = distance;
        distances_count++;
    };

    var closest0:f32 = -1;
    var closest1:f32 = -1;

    for (var i:u32 = 0; i < distances_count; i++) {
        if(distances[i] < closest0) {
            closest1 = closest0;
            closest0 = distances[i];
            continue;
        }
    }

    set_input(closest0, id.x, 2);
    set_input(closest1, id.x, 3);
}

fn prepare_inputs_energy(id: vec3<u32>) {
	let cur_cell = cells[id.x];
	
    set_input(cur_cell.energy, id.x, 4);
}

fn prepare_inputs_memory(id: vec3<u32>) {
	let cur_cell = cells[id.x];
	
    set_input(cur_cell.mem0, id.x, 5);
    set_input(cur_cell.mem1, id.x, 6);
    set_input(cur_cell.mem2, id.x, 7);
    set_input(cur_cell.mem3, id.x, 8);
    set_input(cur_cell.mem4, id.x, 9);
}

fn prepare_inputs(id: vec3<u32>) {
	prepare_inputs_cells(id);
	prepare_inputs_food(id);
	prepare_inputs_energy(id);
	prepare_inputs_memory(id);
}

// ==========INPUTS==========
// 0	= dist to cell0
// 1	= dist to cell1
// 2	= dist to food0
// 3	= dist to food1
// 4	= energy
// 5	= mem0
// 6	= mem1
// 7	= mem2
// 8	= mem3
// 9	= mem4
// ==========================


fn set_input(raw_value: f32, cell: u32, number: u32) {
    inputs[in_id(cell, number)] = tanh(raw_value);
}
fn get_output(cell: u32, number: u32) -> f32 {
    return outputs[out_id(cell, number)];
}


fn calc_neuro(id: vec3<u32>) {
    let cell: u32 = u32(id.x);

    if (cell >= ${cellCount}) {
        return;
    }

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

    let mutation_probability: f32 = 0.05;
    let mutation_strength: f32 = 0.2;

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

    if (global.cells_count >= ${cellCount}u) {
        return;
    }

    if (cells[parent].energy < 0.5) {
        return;
    }

    let child = global.cells_count;

    cells[child] = cells[parent];

    //cells[child].x += 2.0;
    //cells[child].y += 2.0;

    cells[child].energy = cells[parent].energy * 0.5;
    cells[parent].energy *= 0.5;

	cells[child].mem0 = 0f;
	cells[child].mem1 = 0f;
	cells[child].mem2 = 0f;
	cells[child].mem3 = 0f;
	cells[child].mem4 = 0f;

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

	let energy_loss_const = 1f;
	let energy_loss_movement = 2f;

	cells[i].energy -= energy_loss_const + energy_loss_movement * min(1, vlen);

	if(cells[i].energy < 0) {
		kill(id);
		return;
	}

	let act_divide = get_output(i, 2);
	let act_share = get_output(i, 3);
	let act_suicide = get_output(i, 4);
	let act_exist = get_output(i, 5);

	if(act_divide > act_share && act_divide > act_suicide && act_divide > act_exist) {
		divide(id);
	}

	if(act_share > act_divide && act_share > act_suicide && act_share > act_exist) {
		share(id);
	}

	if(act_suicide > act_share && act_suicide > act_divide && act_suicide > act_exist) {
		kill(id);
		return;
	}

	var dmem0 = get_output(i, 6);
	var dmem1 = get_output(i, 7);
	var dmem2 = get_output(i, 8);
	var dmem3 = get_output(i, 9);
	var dmem4 = get_output(i, 10);

	cells[i].mem0 = tanh(cells[i].mem0 * 0.97 + dmem0 * 0.05); // todo set coefficents
	cells[i].mem1 = tanh(cells[i].mem1 * 0.97 + dmem1 * 0.05);
	cells[i].mem2 = tanh(cells[i].mem2 * 0.97 + dmem2 * 0.05);
	cells[i].mem3 = tanh(cells[i].mem3 * 0.97 + dmem3 * 0.05);
	cells[i].mem4 = tanh(cells[i].mem4 * 0.97 + dmem4 * 0.05);
}

// ==========OUTPUTS=========
// 0	= speed X
// 1	= speed Y
//			<<<ACTIONS>>>
// 2	= divide
// 3	= share food
// 4	= suicide
// 5	= exist (do nothing)
//			<<<MEMORY>>>
// 6	= mem0
// 7	= mem1
// 8	= mem2
// 9	= mem3
// 10	= mem4
// ==========================

@compute @workgroup_size(1)
fn tick(@builtin(global_invocation_id) id: vec3<u32>) {
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

		const weightsSize =
			cellCount * cellWeightsVolume * 4

		const inputsSize =
			cellCount * layers[0] * 4

		const outputsSize =
			cellCount * layers[layers.length - 1] * 4

		this.weightsBuffer = device.createBuffer({
			size: weightsSize,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		})

		this.inputsBuffer = device.createBuffer({
			size: inputsSize,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		})

		this.outputsBuffer = device.createBuffer({
			size: outputsSize,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
		})

		this.cellsBuffer = device.createBuffer({
			size: 134217728, //todo change
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
		})

		this.foodBuffer = device.createBuffer({
			size: 134217728, //todo change
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
		})

		this.globalDataBuffer = device.createBuffer({
			size: 12,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
		})

		this.cameraBuffer = device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
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
	}

	tick() {
		const encoder = this.device.createCommandEncoder()
		const pass = encoder.beginComputePass()

		pass.setPipeline(this.pipeline)
		pass.setBindGroup(0, this.bindGroup)

		pass.dispatchWorkgroups(this.cellCount)

		pass.end()

		this.device.queue.submit([encoder.finish()])
	}

	

	loop() {
        console.log('tick')
		this.tick()
		requestAnimationFrame(() => this.loop())
	}
}