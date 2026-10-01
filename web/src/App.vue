<script setup lang="ts">
import { computed, defineAsyncComponent, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  ArrowDown, ArrowLeftRight, ArrowUp, Check, ChevronDown, CircleAlert,
  Download, FileUp, Grip, Pause, Play, Plus, RotateCcw, Sparkles, Square,
  X,
} from 'lucide-vue-next'
import { artifactUrl, cancelJob, createJob, getConfig, getJob, getJobLog } from './api'
import { formatElapsedTime, movePrompt as movePromptInQueue } from './studio-utils'
import type { GenerationOptions, JobSnapshot, PromptItem, StudioConfig } from './types'

const CharacterViewer = defineAsyncComponent(() => import('./components/CharacterViewer.vue'))

const config = ref<StudioConfig | null>(null)
const serviceError = ref('')
const jobError = ref('')
const jobLog = ref('')
const viewerError = ref('')
const file = ref<File | null>(null)
const localUrl = ref<string | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)
const dragActive = ref(false)
const prompts = ref<PromptItem[]>([])
const draft = ref('')
const job = ref<JobSnapshot | null>(null)
const submitting = ref(false)
const playing = ref(false)
const playTime = ref(0)
const seekTime = ref(0)
const seekRevision = ref(0)
const mediaDuration = ref(0)
const selection = ref<number | 'original' | 'sequence'>('original')
const options = ref<GenerationOptions>({
  model: 'preview', mode: 'fast', datasetType: 'objaverse', animMode: 'fk',
  guidance: 3, seed: 42, blendFrames: 8, autoFacing: true,
})
let polling: ReturnType<typeof setInterval> | null = null

const busy = computed(() => submitting.value || job.value?.status === 'running' || job.value?.status === 'queued')
const maxPrompts = 12
const selectedModel = computed(() => config.value?.models.find(item => item.id === options.value.model))
const readyClips = computed(() => job.value?.clips.filter(clip => clip.status === 'ready').length ?? 0)
const timelineItems = computed(() => job.value?.clips ?? prompts.value)
const sequenceReady = computed(() => job.value?.artifacts.includes('sequence_glb') ?? false)
const selectedClip = computed(() => typeof selection.value === 'number' ? job.value?.clips[selection.value] : null)
const source = computed(() => {
  if (job.value && selection.value === 'sequence' && sequenceReady.value) return artifactUrl(job.value.id, 'sequence_glb')
  if (job.value && typeof selection.value === 'number' && job.value.artifacts.includes(`clip_${selection.value + 1}_glb`)) {
    return artifactUrl(job.value.id, `clip_${selection.value + 1}_glb`)
  }
  return localUrl.value
})
const canPlay = computed(() => selection.value !== 'original' && mediaDuration.value > 0)
const selectedTitle = computed(() => {
  if (selection.value === 'sequence') return 'Complete sequence'
  if (typeof selection.value === 'number') return selectedClip.value?.label || `Clip ${selection.value + 1}`
  return file.value?.name || 'Character preview'
})
const selectedSubtitle = computed(() => {
  if (selection.value === 'sequence') return `${job.value?.totalClips ?? 0} clips · ${(job.value?.totalClips ?? 0) * 2} seconds`
  if (typeof selection.value === 'number') return selectedClip.value?.prompt || '2-second motion'
  return file.value ? 'Source rig · orbit to inspect' : 'Your character will appear here'
})

function formatTime(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '—'
  return formatElapsedTime(Math.ceil(seconds))
}

function seek(seconds: number) {
  seekTime.value = seconds
  seekRevision.value++
}

function selectFile(candidate?: File | null) {
  if (!candidate) return
  if (!candidate.name.toLowerCase().endsWith('.glb')) {
    serviceError.value = 'Choose a rigged .glb file. Other formats cannot be uploaded here yet.'
    return
  }
  if (candidate.size > 100 * 1024 * 1024) {
    serviceError.value = 'This file exceeds the 100 MB upload limit.'
    return
  }
  if (localUrl.value) URL.revokeObjectURL(localUrl.value)
  file.value = candidate
  localUrl.value = URL.createObjectURL(candidate)
  selection.value = 'original'
  playTime.value = 0
  seek(0)
  playing.value = false
  viewerError.value = ''
  serviceError.value = ''
  jobError.value = ''
  jobLog.value = ''
  job.value = null
  stopPolling()
}

function handleDrop(event: DragEvent) {
  dragActive.value = false
  if (busy.value) return
  selectFile(event.dataTransfer?.files[0])
}

function addPrompt(text: string, label?: string) {
  const trimmed = text.trim()
  if (!trimmed || prompts.value.length >= maxPrompts || busy.value) return
  prompts.value.push({ text: trimmed, label: label || trimmed.slice(0, 28) })
}

function addDraft() {
  if (!draft.value.trim()) return
  addPrompt(draft.value)
  draft.value = ''
}

function movePrompt(index: number, direction: -1 | 1) {
  if (busy.value) return
  prompts.value = movePromptInQueue(prompts.value, index, direction)
}

function removePrompt(index: number) {
  if (!busy.value) prompts.value.splice(index, 1)
}

function stopPolling() {
  if (polling) clearInterval(polling)
  polling = null
}

async function refreshJob() {
  if (!job.value) return
  try {
    const updated = await getJob(job.value.id)
    job.value = updated
    if (selection.value === 'original' && updated.artifacts.includes('clip_1_glb')) selectView(0)
    if (['ready', 'failed', 'cancelled'].includes(updated.status)) {
      stopPolling()
      if (updated.status === 'failed') void loadLog(updated.id)
    }
  } catch (error) {
    jobError.value = error instanceof Error ? error.message : 'Could not refresh progress'
    stopPolling()
  }
}

async function loadLog(jobId: string) {
  try { jobLog.value = await getJobLog(jobId) }
  catch { jobLog.value = 'The Blender log could not be loaded.' }
}

async function generate() {
  if (!file.value || !prompts.value.length || busy.value || !config.value?.ready) return
  submitting.value = true
  jobError.value = ''
  try {
    const created = await createJob(file.value, prompts.value, options.value)
    selection.value = 'original'
    playing.value = false
    jobLog.value = ''
    job.value = created
    stopPolling()
    polling = setInterval(() => { void refreshJob() }, 1000)
  } catch (error) {
    jobError.value = error instanceof Error ? error.message : 'Could not start generation'
  } finally {
    submitting.value = false
  }
}

async function cancel() {
  if (!job.value || job.value.cancelRequested) return
  try { job.value = await cancelJob(job.value.id) }
  catch (error) { jobError.value = error instanceof Error ? error.message : 'Could not request cancellation' }
}

function selectView(next: number | 'original' | 'sequence') {
  if (next === 'sequence' && !sequenceReady.value) return
  if (typeof next === 'number' && !job.value?.artifacts.includes(`clip_${next + 1}_glb`)) return
  selection.value = next
  playing.value = false
  playTime.value = 0
  seek(0)
  viewerError.value = ''
}

function togglePlay() {
  if (!canPlay.value) return
  if (playTime.value >= mediaDuration.value - 0.01) {
    seek(0)
    playTime.value = 0
  }
  playing.value = !playing.value
}

function scrub(event: Event) {
  const value = Number((event.target as HTMLInputElement).value)
  playing.value = false
  playTime.value = value
  seek(value)
}

function replay() {
  if (!canPlay.value) return
  seek(0)
  playTime.value = 0
  playing.value = true
}

function download(name: string): string {
  return job.value ? artifactUrl(job.value.id, name) : '#'
}

watch(() => options.value.model, () => {
  if (selectedModel.value && !selectedModel.value.available) serviceError.value = 'This checkpoint is not installed locally.'
  else serviceError.value = ''
})

onMounted(async () => {
  try {
    config.value = await getConfig()
    const firstAvailable = config.value.models.find(model => model.available)
    if (firstAvailable) options.value.model = firstAvailable.id
  } catch (error) {
    serviceError.value = error instanceof Error ? error.message : 'Could not reach the local service'
  }
})
onBeforeUnmount(() => {
  stopPolling()
  if (localUrl.value) URL.revokeObjectURL(localUrl.value)
})
</script>

<template>
  <div class="studio-shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark" aria-hidden="true"><span></span><span></span><span></span></span><span>UniMate <strong>Studio</strong></span></div>
      <div class="topbar-meta"><span class="service-dot" :class="{ offline: !config?.ready }"></span>{{ config?.ready ? 'Local renderer ready' : 'Renderer unavailable' }}<span class="topbar-divider"></span><span>2 sec / clip</span></div>
    </header>

    <main class="workspace">
      <aside class="control-rail" aria-label="Animation setup">
        <div class="rail-heading"><div><h1>Motion workspace</h1><p>Give one character a sequence of moves.</p></div><span class="rail-count">{{ prompts.length }}/{{ maxPrompts }}</span></div>

        <section class="rail-section upload-section" aria-labelledby="character-heading">
          <div class="section-heading"><h2 id="character-heading">Character</h2><span>Rigged GLB</span></div>
          <input ref="fileInput" class="visually-hidden" type="file" accept=".glb,model/gltf-binary" @change="selectFile(($event.target as HTMLInputElement).files?.[0])" />
          <button class="upload-zone" :class="{ 'is-dragging': dragActive, 'has-file': file }" type="button" :disabled="busy" @click="fileInput?.click()" @dragenter.prevent="dragActive = true" @dragover.prevent="dragActive = true" @dragleave.prevent="dragActive = false" @drop.prevent="handleDrop">
            <FileUp :size="21" :stroke-width="1.7" aria-hidden="true" />
            <span class="upload-copy"><strong>{{ file ? file.name : 'Drop a rigged character here' }}</strong><small>{{ file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · Choose another GLB` : 'or browse files · GLB up to 100 MB' }}</small></span>
            <span class="upload-action">{{ file ? 'Replace' : 'Browse' }}</span>
          </button>
          <p class="field-hint">Skeleton limit: {{ selectedModel?.maxJoints ?? 61 }} joints. The rig is checked when rendering starts.</p>
        </section>

        <section class="rail-section prompt-section" aria-labelledby="prompts-heading">
          <div class="section-heading"><h2 id="prompts-heading">Motion queue</h2><span>2 seconds each</span></div>
          <p class="section-intro">Add a move, then arrange the order it should play.</p>
          <div class="preset-list" aria-label="Suggested motions">
            <button v-for="preset in config?.presets ?? []" :key="preset.id" class="preset-chip" type="button" :disabled="busy || prompts.length >= maxPrompts" @click="addPrompt(preset.prompt || preset.label, preset.label)"><Plus :size="13" aria-hidden="true" />{{ preset.label }}</button>
          </div>
          <label class="field-label" for="custom-prompt">Write a motion</label>
          <div class="prompt-compose"><textarea id="custom-prompt" v-model="draft" rows="2" maxlength="500" placeholder="A deliberate sidestep, then settles into a ready stance…" :disabled="busy || prompts.length >= maxPrompts" @keydown.meta.enter.prevent="addDraft" @keydown.ctrl.enter.prevent="addDraft"></textarea><button type="button" class="add-prompt" aria-label="Add written motion" :disabled="!draft.trim() || busy || prompts.length >= maxPrompts" @click="addDraft"><Plus :size="18" /></button></div>
          <p class="field-hint">Use a preset as a starting point, or describe your own motion.</p>

          <div v-if="prompts.length" class="queue-list" aria-label="Ordered motion prompts">
            <div v-for="(item, index) in prompts" :key="index" class="queue-item">
              <Grip :size="15" class="grip-icon" aria-hidden="true" />
              <span class="queue-number">{{ String(index + 1).padStart(2, '0') }}</span>
              <div class="queue-content"><input v-model="item.label" :aria-label="`Clip ${index + 1} label`" maxlength="40" :disabled="busy" /><textarea v-model="item.text" :aria-label="`Clip ${index + 1} prompt`" rows="2" maxlength="500" :disabled="busy"></textarea></div>
              <div class="queue-actions"><button type="button" :aria-label="`Move clip ${index + 1} up`" :disabled="busy || index === 0" @click="movePrompt(index, -1)"><ArrowUp :size="14" /></button><button type="button" :aria-label="`Move clip ${index + 1} down`" :disabled="busy || index === prompts.length - 1" @click="movePrompt(index, 1)"><ArrowDown :size="14" /></button><button type="button" :aria-label="`Remove clip ${index + 1}`" :disabled="busy" @click="removePrompt(index)"><X :size="14" /></button></div>
            </div>
          </div>
          <div v-else class="queue-empty">The queue is empty. Pick a move above to begin.</div>
        </section>

        <section class="rail-section settings-section" aria-labelledby="settings-heading">
          <div class="section-heading"><h2 id="settings-heading">Render settings</h2><Sparkles :size="15" aria-hidden="true" /></div>
          <div class="field-grid"><label class="setting-field">Checkpoint<select v-model="options.model" :disabled="busy"><option v-for="model in config?.models ?? []" :key="model.id" :value="model.id" :disabled="!model.available">{{ model.label }}{{ model.available ? '' : ' · missing' }}</option></select></label><label class="setting-field">Seed<input v-model.number="options.seed" type="number" min="0" max="2147483647" :disabled="busy" /></label></div>
          <fieldset class="mode-group" :disabled="busy"><legend>Sampling</legend><label :class="{ selected: options.mode === 'fast' }"><input v-model="options.mode" type="radio" value="fast" /><span><strong>Fast</strong><small>≈11s sampling / clip on M4</small></span></label><label :class="{ selected: options.mode === 'reference' }"><input v-model="options.mode" type="radio" value="reference" /><span><strong>Reference</strong><small>≈59s sampling / clip on M4</small></span></label></fieldset>
          <details class="advanced-settings"><summary>Advanced export <ChevronDown :size="15" aria-hidden="true" /></summary><div class="advanced-body"><label class="setting-field">Rig dataset<select v-model="options.datasetType" :disabled="busy"><option value="objaverse">Objaverse</option><option value="mixamo">Mixamo</option><option value="truebones">Truebones</option></select></label><label class="setting-field">Animation mode<select v-model="options.animMode" :disabled="busy"><option value="fk">Forward kinematics</option><option value="ik">Inverse kinematics</option></select></label><label class="setting-field">Seam blend · {{ options.blendFrames }} frames<input v-model.number="options.blendFrames" type="range" min="0" max="15" :disabled="busy" /></label><label class="setting-field">Guidance · {{ options.guidance }}<input v-model.number="options.guidance" type="range" min="0" max="5" step="0.1" :disabled="busy" /></label></div></details>
        </section>

        <div class="rail-footer"><p v-if="serviceError" class="inline-error" role="alert"><CircleAlert :size="16" />{{ serviceError }}</p><p v-if="jobError" class="inline-error" role="alert"><CircleAlert :size="16" />{{ jobError }}</p><button class="generate-button" type="button" :disabled="!file || !prompts.length || busy || !config?.ready || !selectedModel?.available" @click="generate"><Sparkles :size="17" aria-hidden="true" />{{ submitting ? 'Starting…' : busy ? 'Rendering…' : `Generate ${prompts.length || ''} ${prompts.length === 1 ? 'clip' : 'clips'}` }}</button><p class="footer-note">{{ prompts.length ? `${prompts.length * 2} seconds of animation · ${prompts.length} ${prompts.length === 1 ? 'clip' : 'clips'}` : 'Add a character and at least one motion to generate.' }}</p></div>
      </aside>

      <section class="review-area" aria-label="Character and animation review">
        <div class="review-heading"><div><h2>Preview</h2><p>Orbit, inspect, and play back each take.</p></div><div class="review-status"><span class="status-pip" :class="job?.status || 'idle'"></span>{{ job ? job.status === 'ready' ? 'Sequence ready' : job.status === 'failed' ? 'Render failed' : job.message : file ? 'Character loaded' : 'Waiting for character' }}</div></div>
        <div class="stage"><CharacterViewer :source="source" :playing="playing" :seek-time="seekTime" :seek-revision="seekRevision" @time="playTime = $event" @duration="mediaDuration = $event" @ended="playing = false" @error="viewerError = $event" @loaded="viewerError = ''" /><div class="stage-topline"><span>{{ selectedTitle }}</span><span>{{ selection === 'original' ? 'SOURCE RIG' : selection === 'sequence' ? 'FULL SEQUENCE' : `TAKE ${String(selection + 1).padStart(2, '0')}` }}</span></div><div v-if="!file" class="stage-empty"><div class="empty-mesh" aria-hidden="true"><ArrowLeftRight :size="35" :stroke-width="1" /></div><strong>Start with a character</strong><span>Upload a rigged GLB to see it here.</span></div><div v-if="viewerError" class="stage-error" role="alert">Preview unavailable: {{ viewerError }}</div><div class="stage-caption">{{ selectedSubtitle }}</div></div>
        <div class="transport"><button class="play-button" type="button" :aria-label="playing ? 'Pause animation' : 'Play animation'" :disabled="!canPlay" @click="togglePlay"><Pause v-if="playing" :size="18" fill="currentColor" /><Play v-else :size="18" fill="currentColor" /></button><span class="timecode">{{ formatTime(playTime) }}</span><input class="play-scrubber" type="range" min="0" :max="Math.max(mediaDuration, 0.01)" step="0.01" :value="playTime" :disabled="!canPlay" aria-label="Animation playhead" @input="scrub" /><span class="timecode">{{ formatTime(mediaDuration) }}</span><button class="transport-reset" type="button" aria-label="Replay animation" :disabled="!canPlay" @click="replay"><RotateCcw :size="17" /></button></div>

        <div class="timeline-heading"><div><h2>Sequence</h2><p>Two seconds per move, played in order.</p></div><span class="timeline-total">{{ String(timelineItems.length).padStart(2, '0') }} CLIPS <span>·</span> {{ formatTime(timelineItems.length * 2) }}</span></div>
        <div class="timeline" role="group" aria-label="Animation clip timeline"><button v-for="(item, index) in timelineItems" :key="index" type="button" class="timeline-clip" :class="{ selected: selection === index, ready: job?.clips[index]?.status === 'ready', active: job?.currentClip === index + 1 && busy }" :disabled="!job?.artifacts.includes(`clip_${index + 1}_glb`)" @click="selectView(index)"><span class="timeline-clip-top"><span>{{ String(index + 1).padStart(2, '0') }}</span><Check v-if="job?.clips[index]?.status === 'ready'" :size="15" aria-label="Ready" /><span v-else-if="job?.currentClip === index + 1 && busy" class="tiny-spinner" aria-label="Rendering"></span></span><strong>{{ item.label || `Clip ${index + 1}` }}</strong><small>00:02</small></button><div v-if="!timelineItems.length" class="timeline-placeholder">Your clips will line up here as you add motions.</div></div>

        <div v-if="job" class="job-panel" aria-live="polite"><div class="job-panel-head"><div><strong>{{ job.status === 'ready' ? 'Your sequence is ready' : job.status === 'failed' ? 'Render stopped' : job.status === 'cancelled' ? 'Render cancelled' : job.message }}</strong><span>{{ readyClips }} of {{ job.totalClips }} clips ready <template v-if="job.jointCount">· {{ job.jointCount }} joints</template></span></div><div class="job-panel-clock"><span v-if="busy">{{ formatTime(job.estimatedRemainingSeconds) }} left</span><span v-else>{{ formatTime(job.elapsedSeconds) }} elapsed</span></div></div><div class="progress-track" role="progressbar" :aria-valuenow="Math.round(job.progress * 100)" aria-valuemin="0" aria-valuemax="100" :aria-label="job.message"><span :style="{ transform: `scaleX(${job.progress})` }"></span></div><div class="job-panel-foot"><span>{{ Math.round(job.progress * 100) }}% · {{ job.stage }}</span><button v-if="busy" type="button" :disabled="job.cancelRequested" @click="cancel"><Square :size="12" />{{ job.cancelRequested ? 'Stopping after clip…' : 'Stop after this clip' }}</button></div><p v-if="job.error" class="job-error">{{ job.error }}</p><details v-if="job.status === 'failed' && jobLog" class="job-log"><summary>Recent Blender log</summary><pre>{{ jobLog }}</pre></details></div>

        <div v-if="job && (readyClips || sequenceReady)" class="exports"><div class="exports-head"><h2>Exports</h2><span>GLB for playback · FBX for editing</span></div><div v-if="sequenceReady" class="export-row sequence-export"><button type="button" @click="selectView('sequence')"><span class="export-icon"><ArrowLeftRight :size="18" /></span><span><strong>Complete sequence</strong><small>{{ job.totalClips * 2 }} seconds · {{ job.totalClips * 60 }} frames</small></span></button><div class="export-links"><a :href="download('sequence_glb')" download="unimate-sequence.glb"><Download :size="15" /> GLB</a><a :href="download('sequence_fbx')" download="unimate-sequence.fbx"><Download :size="15" /> FBX</a></div></div><div v-for="clip in job.clips.filter(item => item.status === 'ready')" :key="clip.index" class="export-row"><button type="button" @click="selectView(clip.index)"><span class="export-index">{{ String(clip.index + 1).padStart(2, '0') }}</span><span><strong>{{ clip.label }}</strong><small>2 seconds <template v-if="clip.sampleSeconds">· sampled in {{ clip.sampleSeconds.toFixed(1) }}s</template></small></span></button><div class="export-links"><a :href="download(`clip_${clip.index + 1}_glb`)" :download="`clip-${clip.index + 1}.glb`"><Download :size="15" /> GLB</a><a :href="download(`clip_${clip.index + 1}_fbx`)" :download="`clip-${clip.index + 1}.fbx`"><Download :size="15" /> FBX</a></div></div></div>
      </section>
    </main>
  </div>
</template>
