export interface TriggerUpdateResponse {
  ok: boolean;
  detail?: string;
  message?: string;
}

export function requireSuccessfulTrigger(
  result: TriggerUpdateResponse,
): string {
  if (result.ok !== true) {
    throw new Error(result.detail ?? result.message ?? "Pipeline trigger rejected");
  }
  return result.detail ?? result.message ?? "Triggered successfully";
}
