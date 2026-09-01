export type DependencyStatus = "ok" | (string & {});

export interface HealthResponse {
  status: string;
  service: string;
  env: string;
  dependencies: {
    database: DependencyStatus;
    redis: DependencyStatus;
  };
}
