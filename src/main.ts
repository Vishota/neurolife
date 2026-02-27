import './style.css'
import testNeural from './test'

const canvas = document.createElement('canvas')
canvas.width = window.innerWidth
canvas.height = window.innerHeight

document.body.insertAdjacentElement('beforeend', canvas)


testNeural();