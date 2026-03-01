import Simulation from './Simulation'
import './style.css'

const canvas = document.createElement('canvas')
canvas.width = window.innerWidth
canvas.height = window.innerHeight

document.body.insertAdjacentElement('beforeend', canvas)

const adapter = (await navigator.gpu.requestAdapter())!
const device = (await adapter.requestDevice())!


const sim = new Simulation(device, [5, 5,6, 6], 10000, canvas)

sim.loop()