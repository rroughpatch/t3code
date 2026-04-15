import type {
  ModelCapabilities,
  PiSettings,
  ServerProvider,
  ServerProviderAuth,
  ServerProviderModel,
  ServerProviderSkill,
  ServerProviderSlashCommand,
  ServerProviderState,
} from "@t3tools/contracts";
import { Duration, Effect, Equal, Layer, Schema, Stream } from "effect";
import { ChildProcess, ChildProcessSpawner } from "effect/unstable/process";

import {
  buildServerProvider,
  DEFAULT_TIMEOUT_MS,
  isCommandMissingCause,
  parseGenericCliVersion,
  providerModelsFromSettings,
  spawnAndCollect,
} from "../providerSnapshot";
import { makeManagedServerProvider } from "../makeManagedServerProvider";
import { PiProvider } from "../Services/PiProvider";
import { ServerSettingsService } from "../../serverSettings";
import { ServerConfig } from "../../config";
import { ServerSettingsError } from "@t3tools/contracts";
import { formatPiModelDisplayName, PiRpcClient, toPiModelSlug } from "../piRpc";

const PROVIDER = "pi" as const;
const DEFAULT_PI_MODEL_CAPABILITIES: ModelCapabilities = {
  reasoningEffortLevels: [],
  supportsFastMode: false,
  supportsThinkingToggle: false,
  contextWindowOptions: [],
  promptInjectedEffortLevels: [],
};

type PiAvailableModel = {
  readonly provider: string;
  readonly id: string;
  readonly name?: string;
  readonly reasoning?: boolean;
  readonly contextWindow?: number;
};

type PiCommand = {
  readonly name: string;
  readonly description?: string;
  readonly source: "extension" | "prompt" | "skill";
  readonly sourceInfo: {
    readonly path: string;
    readonly scope?: string;
  };
};

class PiProbeError extends Schema.TaggedErrorClass<PiProbeError>()("PiProbeError", {
  detail: Schema.String,
  cause: Schema.optional(Schema.Defect),
}) {
  override get message(): string {
    return this.detail;
  }
}

function toPiProbeError(detail: string, cause: unknown): PiProbeError {
  return new PiProbeError({
    detail,
    ...(cause !== undefined ? { cause } : {}),
  });
}

function buildPiCapabilities(model: PiAvailableModel): ModelCapabilities {
  return {
    reasoningEffortLevels: model.reasoning
      ? [
          { value: "xhigh", label: "Extra High" },
          { value: "high", label: "High", isDefault: true },
          { value: "medium", label: "Medium" },
          { value: "low", label: "Low" },
        ]
      : [],
    supportsFastMode: false,
    supportsThinkingToggle: false,
    contextWindowOptions:
      typeof model.contextWindow === "number" && Number.isFinite(model.contextWindow)
        ? [
            {
              value: String(model.contextWindow),
              label: String(model.contextWindow),
              isDefault: true,
            },
          ]
        : [],
    promptInjectedEffortLevels: [],
  };
}

function asSlashCommands(
  commands: ReadonlyArray<PiCommand>,
): ReadonlyArray<ServerProviderSlashCommand> {
  return commands
    .filter((command) => command.source !== "skill")
    .map((command) => {
      if (command.description) {
        return {
          name: command.name,
          description: command.description,
        } satisfies ServerProviderSlashCommand;
      }
      return {
        name: command.name,
      } satisfies ServerProviderSlashCommand;
    });
}

function asSkills(commands: ReadonlyArray<PiCommand>): ReadonlyArray<ServerProviderSkill> {
  return commands
    .filter((command) => command.source === "skill")
    .map((command) => {
      const name = command.name.replace(/^skill:/, "");
      if (command.description && command.sourceInfo.scope) {
        return {
          name,
          description: command.description,
          path: command.sourceInfo.path,
          scope: command.sourceInfo.scope,
          enabled: true,
          displayName: name,
        } satisfies ServerProviderSkill;
      }
      if (command.description) {
        return {
          name,
          description: command.description,
          path: command.sourceInfo.path,
          enabled: true,
          displayName: name,
        } satisfies ServerProviderSkill;
      }
      if (command.sourceInfo.scope) {
        return {
          name,
          path: command.sourceInfo.path,
          scope: command.sourceInfo.scope,
          enabled: true,
          displayName: name,
        } satisfies ServerProviderSkill;
      }
      return {
        name,
        path: command.sourceInfo.path,
        enabled: true,
        displayName: name,
      } satisfies ServerProviderSkill;
    });
}

function authSnapshotForModels(models: ReadonlyArray<PiAvailableModel>): {
  readonly status: Exclude<ServerProviderState, "disabled">;
  readonly auth: ServerProviderAuth;
  readonly message?: string;
} {
  if (models.length === 0) {
    return {
      status: "error",
      auth: { status: "unauthenticated" },
      message:
        "Pi is installed but no authenticated models are available. Configure `/login` or API keys for Pi and try again.",
    };
  }

  const providerCount = new Set(models.map((model) => model.provider)).size;
  return {
    status: "ready",
    auth: {
      status: "authenticated",
      type: "configured",
      label:
        providerCount === 1 ? "1 provider configured" : `${providerCount} providers configured`,
    },
  };
}

const runPiCommand = Effect.fn("runPiCommand")(function* (
  binaryPath: string,
  args: ReadonlyArray<string>,
) {
  const command = ChildProcess.make(binaryPath, [...args], {
    shell: process.platform === "win32",
  });
  return yield* spawnAndCollect(binaryPath, command);
});

const probePiCapabilities = Effect.fn("probePiCapabilities")(function* (input: {
  readonly settings: PiSettings;
  readonly cwd: string;
}) {
  const client = new PiRpcClient({
    binaryPath: input.settings.binaryPath,
    cwd: input.cwd,
  });
  yield* Effect.tryPromise({
    try: () => client.start(),
    catch: (cause) => toPiProbeError("Failed to start Pi RPC client.", cause),
  });

  const stop = Effect.tryPromise({
    try: () => client.stop(),
    catch: () => undefined,
  }).pipe(Effect.ignore);

  return yield* Effect.gen(function* () {
    const availableModels = yield* Effect.tryPromise({
      try: () => client.send<{ models: PiAvailableModel[] }>({ type: "get_available_models" }),
      catch: (cause) => toPiProbeError("Failed to query Pi available models.", cause),
    });
    const commands = yield* Effect.tryPromise({
      try: () => client.send<{ commands: PiCommand[] }>({ type: "get_commands" }),
      catch: (cause) => toPiProbeError("Failed to query Pi commands.", cause),
    }).pipe(Effect.orElseSucceed(() => ({ commands: [] as PiCommand[] })));
    return {
      models: availableModels.models,
      commands: commands.commands,
    };
  }).pipe(Effect.ensuring(stop));
});

export const checkPiProviderStatus = Effect.fn("checkPiProviderStatus")(
  function* (): Effect.fn.Return<
    ServerProvider,
    ServerSettingsError,
    ChildProcessSpawner.ChildProcessSpawner | ServerSettingsService | ServerConfig
  > {
    const settings = yield* Effect.service(ServerSettingsService).pipe(
      Effect.flatMap((service) => service.getSettings),
      Effect.map((serverSettings) => serverSettings.providers.pi),
    );
    const { cwd } = yield* ServerConfig;
    const checkedAt = new Date().toISOString();

    if (!settings.enabled) {
      return buildServerProvider({
        provider: PROVIDER,
        enabled: false,
        checkedAt,
        models: [],
        probe: {
          installed: false,
          version: null,
          status: "warning",
          auth: { status: "unknown" },
          message: "Pi is disabled in T3 Code settings.",
        },
      });
    }

    const versionProbe = yield* runPiCommand(settings.binaryPath, ["--version"]).pipe(
      Effect.timeoutOption(DEFAULT_TIMEOUT_MS),
      Effect.result,
    );

    if (versionProbe._tag === "Failure") {
      const error = versionProbe.failure;
      return buildServerProvider({
        provider: PROVIDER,
        enabled: true,
        checkedAt,
        models: providerModelsFromSettings(
          [],
          PROVIDER,
          settings.customModels,
          DEFAULT_PI_MODEL_CAPABILITIES,
        ),
        probe: {
          installed: false,
          version: null,
          status: isCommandMissingCause(error) ? "error" : "warning",
          auth: { status: "unknown" },
          message: isCommandMissingCause(error)
            ? "Pi CLI not found. Install `@mariozechner/pi-coding-agent` or set a custom binary path."
            : error.message,
        },
      });
    }

    const versionResult = versionProbe.success;
    if (versionResult._tag === "None") {
      return buildServerProvider({
        provider: PROVIDER,
        enabled: true,
        checkedAt,
        models: providerModelsFromSettings(
          [],
          PROVIDER,
          settings.customModels,
          DEFAULT_PI_MODEL_CAPABILITIES,
        ),
        probe: {
          installed: true,
          version: null,
          status: "warning",
          auth: { status: "unknown" },
          message: "Timed out while checking Pi CLI version.",
        },
      });
    }

    const version = parseGenericCliVersion(
      `${versionResult.value.stdout}\n${versionResult.value.stderr}`,
    );
    const capabilitiesProbe = yield* probePiCapabilities({ settings, cwd }).pipe(
      Effect.timeoutOption(Duration.seconds(6)),
      Effect.result,
    );

    if (capabilitiesProbe._tag === "Failure") {
      const error = capabilitiesProbe.failure;
      return buildServerProvider({
        provider: PROVIDER,
        enabled: true,
        checkedAt,
        models: providerModelsFromSettings(
          [],
          PROVIDER,
          settings.customModels,
          DEFAULT_PI_MODEL_CAPABILITIES,
        ),
        probe: {
          installed: true,
          version,
          status: "warning",
          auth: { status: "unknown" },
          message: error.message,
        },
      });
    }

    if (capabilitiesProbe.success._tag === "None") {
      return buildServerProvider({
        provider: PROVIDER,
        enabled: true,
        checkedAt,
        models: providerModelsFromSettings(
          [],
          PROVIDER,
          settings.customModels,
          DEFAULT_PI_MODEL_CAPABILITIES,
        ),
        probe: {
          installed: true,
          version,
          status: "warning",
          auth: { status: "unknown" },
          message: "Timed out while querying Pi capabilities.",
        },
      });
    }

    const probed = capabilitiesProbe.success.value;
    const discoveredModels: ReadonlyArray<ServerProviderModel> = probed.models.map((model) => ({
      slug: toPiModelSlug({ provider: model.provider, modelId: model.id }),
      name:
        model.name?.trim() ||
        formatPiModelDisplayName({ provider: model.provider, modelId: model.id }),
      isCustom: false,
      capabilities: buildPiCapabilities(model),
    }));

    const models = providerModelsFromSettings(
      discoveredModels,
      PROVIDER,
      settings.customModels,
      DEFAULT_PI_MODEL_CAPABILITIES,
    );
    const auth = authSnapshotForModels(probed.models);

    return buildServerProvider({
      provider: PROVIDER,
      enabled: true,
      checkedAt,
      models,
      slashCommands: asSlashCommands(probed.commands),
      skills: asSkills(probed.commands),
      probe: {
        installed: true,
        version,
        status: auth.status,
        auth: auth.auth,
        ...(auth.message ? { message: auth.message } : {}),
      },
    });
  },
);

const makePendingPiProvider = (piSettings: PiSettings): ServerProvider =>
  buildServerProvider({
    provider: PROVIDER,
    enabled: piSettings.enabled,
    checkedAt: new Date(0).toISOString(),
    models: providerModelsFromSettings(
      [],
      PROVIDER,
      piSettings.customModels,
      DEFAULT_PI_MODEL_CAPABILITIES,
    ),
    probe: {
      installed: false,
      version: null,
      status: "warning",
      auth: { status: "unknown" },
      message: "Checking Pi status...",
    },
  });

export const PiProviderLive = Layer.effect(
  PiProvider,
  Effect.gen(function* () {
    const serverSettings = yield* ServerSettingsService;
    const serverConfig = yield* ServerConfig;
    const spawner = yield* ChildProcessSpawner.ChildProcessSpawner;

    const checkProvider = checkPiProviderStatus().pipe(
      Effect.provideService(ServerSettingsService, serverSettings),
      Effect.provideService(ServerConfig, serverConfig),
      Effect.provideService(ChildProcessSpawner.ChildProcessSpawner, spawner),
    );

    return yield* makeManagedServerProvider<PiSettings>({
      getSettings: serverSettings.getSettings.pipe(
        Effect.map((settings) => settings.providers.pi),
        Effect.orDie,
      ),
      streamSettings: serverSettings.streamChanges.pipe(
        Stream.map((settings) => settings.providers.pi),
      ),
      haveSettingsChanged: (previous, next) => !Equal.equals(previous, next),
      initialSnapshot: makePendingPiProvider,
      checkProvider,
    });
  }),
);
