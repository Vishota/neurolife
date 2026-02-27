import Neural from "./Neural";

export default async function testNeural() {
    if (!navigator.gpu) {
        console.error("WebGPU not supported!");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter found");

    const device = await adapter.requestDevice();

    // Network: 4 inputs → 8 hidden → 2 outputs
    const layers = [200, 200, 200, 200];
    const cellCount = 1;

    // Create Neural instance
    const neural = new Neural(device, layers, cellCount);

    // Fill weights with random numbers
    const cellWeightsVolume = layers
        .slice(0, -1)
        .reduce((sum, layer, i) => sum + layer * layers[i + 1], 0);

    const weightsArray = new Float32Array(cellCount * cellWeightsVolume);
    for (let i = 0; i < weightsArray.length; i++) {
        weightsArray[i] = Math.random() * 2 - 1; // [-1,1]
    }

    device.queue.writeBuffer(
        neural.weightsBuffer,
        0,
        weightsArray.buffer,
        weightsArray.byteOffset,
        weightsArray.byteLength
    );

    // Fill inputs with random numbers
    const inputsArray = new Float32Array(cellCount * layers[0]);
    for (let i = 0; i < inputsArray.length; i++) {
        inputsArray[i] = Math.random();
    }

    device.queue.writeBuffer(
        neural.inputsBuffer,
        0,
        inputsArray.buffer,
        inputsArray.byteOffset,
        inputsArray.byteLength
    );

    // Run one tick
    neural.tick();

    // Read back outputs for testing
    const outputsSize = cellCount * layers[layers.length - 1] * 4;
    const readBuffer = device.createBuffer({
        size: outputsSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(
        neural.outputsBuffer,
        0,
        readBuffer,
        0,
        outputsSize
    );
    device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const outputArray = new Float32Array(readBuffer.getMappedRange().slice(0));
    console.log("Outputs per cell:");
    for (let c = 0; c < cellCount; c++) {
        const outStart = c * layers[layers.length - 1];
        const out = outputArray.slice(outStart, outStart + layers[layers.length - 1]);
        console.log(`Cell ${c}:`, Array.from(out));
    }

    readBuffer.unmap();
}
