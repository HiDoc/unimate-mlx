export interface ModelOption {
  id: string;
  label: string;
  maxJoints: number;
  available: boolean;
}

export interface PromptItem {
  text: string;
  label?: string;
}

export interface PromptPreset {
  id: string;
  label: string;
  prompt?: string;
}

export interface StudioConfig {
  ready: boolean;
  referenceAvailable: boolean;
  referencePath: string;
  models: ModelOption[];
  presets: PromptPreset[];
  clipSeconds: number;
  framesPerClip: number;
  framesPerSecond: number;
  timingNote: string;
}

export interface ClipSnapshot {
  index: number;
  label: string;
  prompt: string;
  status: "queued" | "generating" | "ready" | "failed" | string;
  durationSeconds: number;
  seed: number;
  sampleSeconds?: number;
}

export interface JobSnapshot {
  id: string;
  filename: string;
  status: "queued" | "running" | "ready" | "failed" | "cancelled" | string;
  stage: string;
  message: string;
  error: string | null;
  progress: number;
  currentClip: number;
  totalClips: number;
  clips: ClipSnapshot[];
  artifacts: string[];
  jointCount: number | null;
  estimatedRemainingSeconds: number | null;
  stageEstimateSeconds: number;
  elapsedSeconds: number;
  sampleSeconds: number[];
  cancelRequested: boolean;
  createdAt: number;
}

export interface GenerationOptions {
  model: string;
  mode: "fast" | "reference";
  datasetType: "objaverse" | "mixamo" | "truebones";
  animMode: "fk" | "ik";
  guidance: number;
  seed: number;
  blendFrames: number;
  faceRight?: string;
  faceLeft?: string;
  autoFacing?: boolean;
}
