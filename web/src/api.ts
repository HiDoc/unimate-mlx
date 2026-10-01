import type {
  GenerationOptions,
  JobSnapshot,
  PromptItem,
  StudioConfig,
} from "./types";

const API_ROOT = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_ROOT}${path}`, init);
  } catch (error) {
    const reason = error instanceof Error ? error.message : "Network request failed";
    throw new Error(`Could not reach UniMate Studio: ${reason}`);
  }

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`.trim();
    try {
      const body: unknown = await response.json();
      if (typeof body === "object" && body !== null && "detail" in body) {
        const detail = (body as { detail: unknown }).detail;
        if (typeof detail === "string") message = detail;
        else if (Array.isArray(detail)) {
          message = detail
            .map((item) => {
              if (typeof item === "object" && item !== null && "msg" in item) {
                return String((item as { msg: unknown }).msg);
              }
              return String(item);
            })
            .join("; ");
        }
      }
    } catch {
      // Keep the HTTP status message when the server does not return JSON.
    }
    throw new Error(message || "The request failed");
  }

  return (await response.json()) as T;
}

export function getConfig(): Promise<StudioConfig> {
  return request<StudioConfig>("/config");
}

export function getJob(jobId: string): Promise<JobSnapshot> {
  return request<JobSnapshot>(`/jobs/${encodeURIComponent(jobId)}`);
}
export async function getJobLog(jobId: string): Promise<string> {
  const result = await request<{ text: string }>(`/jobs/${encodeURIComponent(jobId)}/log`);
  return result.text;
}

export function listJobs(): Promise<JobSnapshot[]> {
  return request<JobSnapshot[]>("/jobs");
}

export function createJob(
  file: File,
  prompts: PromptItem[],
  options: GenerationOptions,
): Promise<JobSnapshot> {
  const form = new FormData();
  form.append("character", file, file.name);
  form.append("payload", JSON.stringify({ prompts, ...options }));
  return request<JobSnapshot>("/jobs", { method: "POST", body: form });
}

export function cancelJob(jobId: string): Promise<JobSnapshot> {
  return request<JobSnapshot>(`/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });
}

export function artifactUrl(jobId: string, artifact: string): string {
  return `${API_ROOT}/jobs/${encodeURIComponent(jobId)}/files/${encodeURIComponent(artifact)}`;
}
