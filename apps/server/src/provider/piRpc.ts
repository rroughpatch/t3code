import { randomUUID } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";

export interface PiModelRef {
  readonly provider: string;
  readonly modelId: string;
}

export interface PiRpcResponse {
  readonly id?: string;
  readonly type: "response";
  readonly command: string;
  readonly success: boolean;
  readonly data?: unknown;
  readonly error?: string;
}

export interface PiRpcEventRecord {
  readonly type: string;
  readonly [key: string]: unknown;
}

export type PiRpcRecord = PiRpcResponse | PiRpcEventRecord;

export interface PiRpcClientOptions {
  readonly binaryPath: string;
  readonly cwd: string;
  readonly sessionFile?: string;
  readonly initialModel?: PiModelRef;
  readonly env?: NodeJS.ProcessEnv;
  readonly extraArgs?: ReadonlyArray<string>;
  readonly onRecord?: (record: PiRpcEventRecord) => void;
  readonly onExit?: (input: {
    readonly code: number | null;
    readonly signal: NodeJS.Signals | null;
    readonly stderr: string;
  }) => void;
}

function isResponse(value: unknown): value is PiRpcResponse {
  return (
    typeof value === "object" &&
    value !== null &&
    "type" in value &&
    (value as { type?: unknown }).type === "response"
  );
}

function rejectPending(
  pending: Map<string, { reject: (error: Error) => void; timeout: ReturnType<typeof setTimeout> }>,
  error: Error,
) {
  for (const { reject, timeout } of pending.values()) {
    clearTimeout(timeout);
    reject(error);
  }
  pending.clear();
}

export function parsePiModelSlug(value: string | null | undefined): PiModelRef | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  const separatorIndex = trimmed.indexOf("/");
  if (separatorIndex <= 0 || separatorIndex >= trimmed.length - 1) {
    return undefined;
  }
  const provider = trimmed.slice(0, separatorIndex).trim();
  const modelId = trimmed.slice(separatorIndex + 1).trim();
  if (!provider || !modelId) {
    return undefined;
  }
  return { provider, modelId };
}

export function toPiModelSlug(input: PiModelRef): string {
  return `${input.provider}/${input.modelId}`;
}

function titleCase(value: string): string {
  return value
    .split(/[\s/_-]+/g)
    .filter(Boolean)
    .map((part) => part[0]!.toUpperCase() + part.slice(1))
    .join(" ");
}

export function formatPiModelDisplayName(input: PiModelRef): string {
  return `${titleCase(input.provider)} · ${input.modelId}`;
}

export class PiRpcClient {
  private process: ChildProcessWithoutNullStreams | null = null;
  private stdoutBuffer = "";
  private stderr = "";
  private readonly pending = new Map<
    string,
    {
      resolve: (response: PiRpcResponse) => void;
      reject: (error: Error) => void;
      timeout: ReturnType<typeof setTimeout>;
    }
  >();

  constructor(private readonly options: PiRpcClientOptions) {}

  async start(): Promise<void> {
    if (this.process) {
      throw new Error("Pi RPC client already started.");
    }

    if (this.options.sessionFile) {
      await mkdir(dirname(this.options.sessionFile), { recursive: true });
    }

    const args = ["--mode", "rpc", "--no-extension"];
    if (this.options.sessionFile) {
      args.push(
        "--session",
        this.options.sessionFile,
        "--session-dir",
        dirname(this.options.sessionFile),
      );
    } else {
      args.push("--no-session");
    }
    if (this.options.initialModel) {
      args.push(
        "--provider",
        this.options.initialModel.provider,
        "--model",
        this.options.initialModel.modelId,
      );
    }
    if (this.options.extraArgs) {
      args.push(...this.options.extraArgs);
    }

    const child = spawn(this.options.binaryPath, args, {
      cwd: this.options.cwd,
      env: { ...process.env, ...this.options.env },
      shell: process.platform === "win32",
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.process = child;

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      this.stdoutBuffer += chunk;
      let newlineIndex = this.stdoutBuffer.indexOf("\n");
      while (newlineIndex >= 0) {
        let line = this.stdoutBuffer.slice(0, newlineIndex);
        this.stdoutBuffer = this.stdoutBuffer.slice(newlineIndex + 1);
        if (line.endsWith("\r")) {
          line = line.slice(0, -1);
        }
        this.handleLine(line);
        newlineIndex = this.stdoutBuffer.indexOf("\n");
      }
    });

    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      this.stderr += chunk;
    });

    child.on("exit", (code, signal) => {
      const error = new Error(
        `Pi RPC process exited${code !== null ? ` with code ${code}` : ""}${signal ? ` via ${signal}` : ""}.`,
      );
      rejectPending(this.pending, error);
      this.options.onExit?.({ code, signal, stderr: this.stderr });
      this.process = null;
    });

    await new Promise((resolve) => setTimeout(resolve, 100));
    if (child.exitCode !== null) {
      throw new Error(
        `Pi RPC process exited immediately with code ${child.exitCode}.${this.stderr ? ` ${this.stderr.trim()}` : ""}`,
      );
    }
  }

  async stop(): Promise<void> {
    const child = this.process;
    if (!child) {
      return;
    }

    child.kill("SIGTERM");
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        resolve();
      }, 1_000);
      child.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
    });
    this.process = null;
  }

  async send<TData = unknown>(
    command: Omit<Record<string, unknown>, "id"> & { readonly type: string },
    timeoutMs = 8_000,
  ): Promise<TData> {
    const child = this.process;
    if (!child) {
      throw new Error("Pi RPC client is not running.");
    }

    const id = randomUUID();
    const payload = JSON.stringify({ ...command, id });
    const responsePromise = new Promise<PiRpcResponse>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Pi RPC command timed out: ${command.type}`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timeout });
    });

    child.stdin.write(`${payload}\n`);
    const response = await responsePromise;
    if (!response.success) {
      throw new Error(response.error || `Pi RPC command failed: ${command.type}`);
    }
    return response.data as TData;
  }

  getStderr(): string {
    return this.stderr;
  }

  private handleLine(line: string) {
    if (!line.trim()) {
      return;
    }

    let parsed: PiRpcRecord;
    try {
      parsed = JSON.parse(line) as PiRpcRecord;
    } catch {
      return;
    }

    if (isResponse(parsed)) {
      if (parsed.id) {
        const pending = this.pending.get(parsed.id);
        if (pending) {
          clearTimeout(pending.timeout);
          this.pending.delete(parsed.id);
          pending.resolve(parsed);
        }
      }
      return;
    }

    this.options.onRecord?.(parsed);
  }
}
