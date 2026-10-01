<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

const props = defineProps<{ source: string | null; playing: boolean; seekTime: number; seekRevision: number }>()
const emit = defineEmits<{
  time: [seconds: number]
  duration: [seconds: number]
  ended: []
  error: [message: string]
  loaded: []
}>()

const host = ref<HTMLDivElement | null>(null)
const loading = ref(false)
let renderer: THREE.WebGLRenderer | null = null
let scene: THREE.Scene
let camera: THREE.PerspectiveCamera
let controls: OrbitControls
let mixer: THREE.AnimationMixer | null = null
let action: THREE.AnimationAction | null = null
let model: THREE.Object3D | null = null
let duration = 0
let currentTime = 0
let animationFrame = 0
let previousFrame = 0
let resizeObserver: ResizeObserver | null = null
let loadVersion = 0

function resize() {
  if (!host.value || !renderer) return
  const width = host.value.clientWidth
  const height = host.value.clientHeight
  if (!width || !height) return
  camera.aspect = width / height
  camera.updateProjectionMatrix()
  renderer.setSize(width, height, false)
}

function frame(now: number) {
  animationFrame = requestAnimationFrame(frame)
  const delta = Math.min((now - previousFrame) / 1000, 0.1)
  previousFrame = now
  if (props.playing && mixer && duration > 0) {
    currentTime += delta
    if (currentTime >= duration) {
      currentTime = duration
      emit('ended')
    }
    mixer.setTime(currentTime)
    emit('time', currentTime)
  }
  controls?.update()
  renderer?.render(scene, camera)
}

function clearModel() {
  if (model) {
    scene.remove(model)
    model.traverse((part) => {
      if (part instanceof THREE.Mesh) {
        part.geometry.dispose()
        const materials = Array.isArray(part.material) ? part.material : [part.material]
        for (const material of materials) material.dispose()
      }
    })
  }
  model = null
  mixer = null
  action = null
  duration = 0
  currentTime = 0
  emit('duration', 0)
}

async function load(source: string | null) {
  const version = ++loadVersion
  if (!scene) return
  clearModel()
  if (!source) return
  loading.value = true
  try {
    const gltf = await new GLTFLoader().loadAsync(source)
    if (version !== loadVersion) return
    model = gltf.scene
    scene.add(model)
    const box = new THREE.Box3().setFromObject(model)
    if (!box.isEmpty()) {
      const size = box.getSize(new THREE.Vector3())
      const center = box.getCenter(new THREE.Vector3())
      const radius = Math.max(size.length() / 2, 0.5)
      const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.12
      controls.target.copy(center)
      camera.position.copy(center).add(new THREE.Vector3(distance * 0.66, distance * 0.35, distance))
      camera.near = Math.max(0.01, distance / 100)
      camera.far = Math.max(100, distance * 20)
      camera.updateProjectionMatrix()
      controls.minDistance = radius * 0.5
      controls.maxDistance = distance * 5
      controls.update()
    }
    if (gltf.animations.length) {
      mixer = new THREE.AnimationMixer(model)
      const clip = gltf.animations[0]
      duration = clip.duration
      action = mixer.clipAction(clip)
      action.setLoop(THREE.LoopOnce, 1)
      action.clampWhenFinished = true
      action.play()
      mixer.setTime(0)
    }
    emit('duration', duration)
    emit('loaded')
  } catch (error) {
    if (version === loadVersion) emit('error', error instanceof Error ? error.message : 'Could not open this GLB')
  } finally {
    if (version === loadVersion) loading.value = false
  }
}

watch(() => props.source, load)
watch(() => [props.seekTime, props.seekRevision] as const, ([time]) => {
  currentTime = THREE.MathUtils.clamp(time, 0, duration)
  mixer?.setTime(currentTime)
})

onMounted(() => {
  if (!host.value) return
  scene = new THREE.Scene()
  scene.background = new THREE.Color('#192120')
  camera = new THREE.PerspectiveCamera(38, 1, 0.01, 200)
  camera.position.set(3, 2, 5)
  scene.add(new THREE.HemisphereLight(0xeef5f0, 0x566964, 2.5))
  const key = new THREE.DirectionalLight(0xffffff, 3)
  key.position.set(4, 7, 6)
  scene.add(key)
  const rim = new THREE.DirectionalLight(0xa8c6c0, 2)
  rim.position.set(-4, 3, -5)
  scene.add(rim)
  const grid = new THREE.GridHelper(24, 24, 0x40524d, 0x283632)
  grid.position.y = -0.01
  scene.add(grid)
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
  } catch {
    emit('error', 'This browser could not start WebGL. Generation and downloads are still available.')
    return
  }
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  host.value.appendChild(renderer.domElement)
  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  resizeObserver = new ResizeObserver(resize)
  resizeObserver.observe(host.value)
  resize()
  previousFrame = performance.now()
  animationFrame = requestAnimationFrame(frame)
  void load(props.source)
})

onBeforeUnmount(() => {
  loadVersion++
  cancelAnimationFrame(animationFrame)
  resizeObserver?.disconnect()
  controls?.dispose()
  clearModel()
  renderer?.dispose()
  renderer?.domElement.remove()
})
</script>

<template>
  <div ref="host" class="viewer-canvas" aria-label="3D character preview">
    <div v-if="loading" class="viewer-loading" role="status">Opening character…</div>
  </div>
</template>
