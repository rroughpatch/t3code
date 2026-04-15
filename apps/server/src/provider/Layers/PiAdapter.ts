import { randomUUID } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

import {
  type CanonicalItemType,
  EventId,
  type ProviderRuntimeEvent,
  type ProviderSendTurnInput,
  type ProviderSession,
  ProviderItemId,
  type ProviderUserInputAnswers,
  RuntimeItemId,
  ThreadId,
  TurnId,
  type ThreadTokenUsageSnapshot,
  type PiSettings,
} from "@t3tools/contracts";
import { Effect, Layer, Queue, Schema, Stream } from "effect";

import { resolveAttachmentPath } from "../../attachmentStore.ts";
import { toSafeThreadAttachmentSegment } from "../../attachmentStore.ts";
import { ServerConfig } from "../../config.ts";
import { ServerSettingsService } from "../../serverSettings.ts";
import { type EventNdjsonLogger, makeEventNdjsonLogger } from "./EventNdjsonLogger.ts";
import {
  ProviderAdapterProcessError,
  ProviderAdapterRequestError,
  ProviderAdapterSessionNotFoundError,
  ProviderAdapterValidationError,
} from "../Errors.ts";
import { type ProviderThreadSnapshot } from "../Services/ProviderAdapter.ts";
import { PiAdapter, type PiAdapterShape } from "../Services/PiAdapter.ts";
import { parsePiModelSlug, PiRpcClient, toPiModelSlug, type PiRpcEventRecord } from "../piRpc.ts";

const PROVIDER = "pi" as const;

type PiMessage = {
  readonly role?: string;
  readonly content?: unknown;
  readonly stopReason?: string;
  readonly errorMessage?: string;
  readonly usage?: {
    readonly input?: number;
    readonly output?: number;
    readonly cacheRead?: number;
    readonly cacheWrite?: number;
    readonly totalTokens?: number;
  };
};

interface PiTurnState {
  readonly turnId: TurnId;
  readonly assistantItemId: RuntimeItemId;
  readonly startedAt: string;
  readonly items: Array<unknown>;
  assistantText: string;
  readonly toolItemIdsByCallId: Map<string, RuntimeItemId>;
}

interface PiSessionContext {
  session: ProviderSession;
  readonly client: PiRpcClient;
  readonly sessionFile: string;
  readonly turns: Array<{ id: TurnId; items: Array<unknown> }>;
  currentModel: string | undefined;
  turnState: PiTurnState | undefined;
  nextTurnId: TurnId | undefined;
  stopped: boolean;
}

type PiInputAttachment = NonNullable<ProviderSendTurnInput["attachments"]>[number];

export interface PiAdapterLiveOptions {
  readonly nativeEventLogPath?: string;
  readonly nativeEventLogger?: EventNdjsonLogger;
}

function nowIso(): string {
  return new Date().toISOString();
}

function makeEventId(): EventId {
  return EventId.make(`pi-event-${randomUUID()}`);
}

function makeTurnId(): TurnId {
  return TurnId.make(`pi-turn-${randomUUID()}`);
}

function makeItemId(prefix: string): RuntimeItemId {
  return RuntimeItemId.make(`pi-${prefix}-${randomUUID()}`);
}

function textFromContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  return content
    .flatMap((entry) => {
      if (!entry || typeof entry !== "object") {
        return [];
      }
      const record = entry as Record<string, unknown>;
      if (record.type === "text" && typeof record.text === "string") {
        return [record.text];
      }
      if (record.type === "thinking" && typeof record.thinking === "string") {
        return [record.thinking];
      }
      return [];
    })
    .join("");
}

function turnStatusFromMessage(
  message: PiMessage | undefined,
): "completed" | "failed" | "interrupted" | "cancelled" {
  switch (message?.stopReason) {
    case "aborted":
      return "interrupted";
    case "error":
      return "failed";
    default:
      return "completed";
  }
}

function toolItemType(toolName: string): CanonicalItemType {
  switch (toolName) {
    case "bash":
      return "command_execution";
    case "write":
    case "edit":
      return "file_change";
    case "web_search":
      return "web_search";
    case "image_view":
      return "image_view";
    default:
      return "dynamic_tool_call";
  }
}

function usageFromMessage(message: PiMessage | undefined): ThreadTokenUsageSnapshot | undefined {
  const usage = message?.usage;
  if (!usage || typeof usage.totalTokens !== "number" || usage.totalTokens <= 0) {
    return undefined;
  }
  return {
    usedTokens: usage.totalTokens,
    totalProcessedTokens: usage.totalTokens,
    ...(typeof usage.input === "number"
      ? { inputTokens: usage.input, lastInputTokens: usage.input }
      : {}),
    ...(typeof usage.output === "number"
      ? { outputTokens: usage.output, lastOutputTokens: usage.output }
      : {}),
    ...(typeof usage.cacheRead === "number"
      ? { cachedInputTokens: usage.cacheRead, lastCachedInputTokens: usage.cacheRead }
      : {}),
    lastUsedTokens: usage.totalTokens,
  };
}

function groupTurnsFromMessages(
  messages: ReadonlyArray<unknown>,
  existing: ReadonlyArray<{ id: TurnId; items: Array<unknown> }>,
): Array<{ id: TurnId; items: Array<unknown> }> {
  const grouped: Array<Array<unknown>> = [];
  let current: Array<unknown> | undefined;

  for (const message of messages) {
    const role =
      typeof message === "object" && message !== null ? (message as PiMessage).role : undefined;
    if (role === "user") {
      current = [message];
      grouped.push(current);
      continue;
    }
    if (!current) {
      current = [message];
      grouped.push(current);
      continue;
    }
    current.push(message);
  }

  return grouped.map((items, index) => ({
    id: existing[index]?.id ?? TurnId.make(`pi-history-turn-${index + 1}`),
    items,
  }));
}

function toValidationError(operation: string, issue: string, cause?: unknown) {
  return new ProviderAdapterValidationError({
    provider: PROVIDER,
    operation,
    issue,
    ...(cause !== undefined ? { cause } : {}),
  });
}

async function readAttachmentAsPiImage(input: {
  readonly attachmentsDir: string;
  readonly attachment: PiInputAttachment;
}) {
  const filePath = resolveAttachmentPath({
    attachmentsDir: input.attachmentsDir,
    attachment: input.attachment,
  });
  if (!filePath) {
    throw new Error(`Attachment not found: ${input.attachment.id}`);
  }
  const bytes = await readFile(filePath);
  return {
    type: "image" as const,
    data: bytes.toString("base64"),
    mimeType: input.attachment.mimeType,
  };
}

const makePiAdapter = Effect.fn("makePiAdapter")(function* (options?: PiAdapterLiveOptions) {
  const config = yield* ServerConfig;
  const serverSettings = yield* ServerSettingsService;
  const nativeEventLogger =
    options?.nativeEventLogger ??
    (options?.nativeEventLogPath !== undefined
      ? yield* makeEventNdjsonLogger(options.nativeEventLogPath, { stream: "native" })
      : undefined);
  const runtimeEvents = yield* Queue.unbounded<ProviderRuntimeEvent>();
  const sessions = new Map<ThreadId, PiSessionContext>();

  const emit = (event: ProviderRuntimeEvent, raw?: PiRpcEventRecord, threadId?: ThreadId) =>
    Effect.gen(function* () {
      if (raw && nativeEventLogger) {
        yield* nativeEventLogger
          .write(raw, threadId ?? event.threadId)
          .pipe(Effect.ignore({ log: true }));
      }
      yield* Queue.offer(runtimeEvents, event);
    });

  const requireSession = (threadId: ThreadId): PiSessionContext => {
    const session = sessions.get(threadId);
    if (!session) {
      throw new ProviderAdapterSessionNotFoundError({ provider: PROVIDER, threadId });
    }
    return session;
  };

  const getPiSettings = (threadId: ThreadId, operation: string) =>
    serverSettings.getSettings.pipe(
      Effect.map((settings) => settings.providers.pi),
      Effect.mapError(
        (error) =>
          new ProviderAdapterProcessError({
            provider: PROVIDER,
            threadId,
            detail: `Failed to read Pi settings while handling ${operation}: ${error.message}`,
            cause: error,
          }),
      ),
    );

  const sessionFilePath = (threadId: ThreadId) => {
    const segment = toSafeThreadAttachmentSegment(threadId) ?? threadId;
    return join(config.providerLogsDir, "pi-sessions", `${segment}.jsonl`);
  };

  const rebuildTurns = async (session: PiSessionContext) => {
    const messages = await session.client.send<{ messages: unknown[] }>({ type: "get_messages" });
    session.turns.splice(
      0,
      session.turns.length,
      ...groupTurnsFromMessages(messages.messages, session.turns),
    );
    return {
      threadId: session.session.threadId,
      turns: session.turns.map((turn) => ({ id: turn.id, items: [...turn.items] })),
    } satisfies ProviderThreadSnapshot;
  };

  const handleUnexpectedExit = (
    session: PiSessionContext,
    input: { code: number | null; signal: NodeJS.Signals | null; stderr: string },
  ) =>
    Effect.gen(function* () {
      if (session.stopped) {
        return;
      }
      sessions.delete(session.session.threadId);
      const createdAt = nowIso();
      yield* emit(
        {
          type: "runtime.error",
          eventId: makeEventId(),
          provider: PROVIDER,
          threadId: session.session.threadId,
          createdAt,
          payload: {
            message:
              input.stderr.trim() ||
              `Pi process exited unexpectedly${input.code !== null ? ` (code ${input.code})` : ""}.`,
            class: "provider_error",
          },
        },
        undefined,
      );
      yield* emit({
        type: "session.state.changed",
        eventId: makeEventId(),
        provider: PROVIDER,
        threadId: session.session.threadId,
        createdAt,
        payload: {
          state: "error",
          reason: input.stderr.trim() || "Pi process exited unexpectedly.",
        },
      });
      yield* emit({
        type: "session.exited",
        eventId: makeEventId(),
        provider: PROVIDER,
        threadId: session.session.threadId,
        createdAt,
        payload: {
          exitKind: "error",
          reason: input.signal ?? (input.code !== null ? `code ${input.code}` : "unknown"),
          recoverable: true,
        },
      });
    }).pipe(Effect.runPromise);

  const handleRecord = (session: PiSessionContext, raw: PiRpcEventRecord) =>
    Effect.gen(function* () {
      const createdAt = nowIso();
      switch (raw.type) {
        case "turn_start": {
          const turnId = session.nextTurnId ?? makeTurnId();
          session.nextTurnId = undefined;
          session.turnState = {
            turnId,
            assistantItemId: makeItemId("assistant"),
            startedAt: createdAt,
            items: [],
            assistantText: "",
            toolItemIdsByCallId: new Map(),
          };
          yield* emit(
            {
              type: "session.state.changed",
              eventId: makeEventId(),
              provider: PROVIDER,
              threadId: session.session.threadId,
              createdAt,
              payload: { state: "running" },
            },
            raw,
            session.session.threadId,
          );
          yield* emit(
            {
              type: "turn.started",
              eventId: makeEventId(),
              provider: PROVIDER,
              threadId: session.session.threadId,
              turnId,
              createdAt,
              payload: {},
            },
            raw,
            session.session.threadId,
          );
          break;
        }

        case "message_update": {
          if (session.turnState === undefined) {
            break;
          }
          const message = raw.message as PiMessage | undefined;
          const assistantEvent =
            raw.assistantMessageEvent && typeof raw.assistantMessageEvent === "object"
              ? (raw.assistantMessageEvent as Record<string, unknown>)
              : undefined;
          if (message?.role !== "assistant" || !assistantEvent) {
            break;
          }
          if (
            assistantEvent.type === "text_delta" &&
            typeof assistantEvent.delta === "string" &&
            assistantEvent.delta.length > 0
          ) {
            session.turnState.assistantText += assistantEvent.delta;
            yield* emit(
              {
                type: "content.delta",
                eventId: makeEventId(),
                provider: PROVIDER,
                threadId: session.session.threadId,
                turnId: session.turnState.turnId,
                itemId: session.turnState.assistantItemId,
                createdAt,
                payload: {
                  streamKind: "assistant_text",
                  delta: assistantEvent.delta,
                },
              },
              raw,
              session.session.threadId,
            );
          }
          if (
            assistantEvent.type === "thinking_delta" &&
            typeof assistantEvent.delta === "string" &&
            assistantEvent.delta.length > 0
          ) {
            yield* emit(
              {
                type: "content.delta",
                eventId: makeEventId(),
                provider: PROVIDER,
                threadId: session.session.threadId,
                turnId: session.turnState.turnId,
                itemId: session.turnState.assistantItemId,
                createdAt,
                payload: {
                  streamKind: "reasoning_text",
                  delta: assistantEvent.delta,
                },
              },
              raw,
              session.session.threadId,
            );
          }
          break;
        }

        case "tool_execution_start": {
          if (
            session.turnState === undefined ||
            typeof raw.toolCallId !== "string" ||
            typeof raw.toolName !== "string"
          ) {
            break;
          }
          const itemId = makeItemId(raw.toolName);
          session.turnState.toolItemIdsByCallId.set(raw.toolCallId, itemId);
          yield* emit(
            {
              type: "item.started",
              eventId: makeEventId(),
              provider: PROVIDER,
              threadId: session.session.threadId,
              turnId: session.turnState.turnId,
              itemId,
              createdAt,
              providerRefs: { providerItemId: ProviderItemId.make(raw.toolCallId) },
              payload: {
                itemType: toolItemType(raw.toolName),
                title: raw.toolName,
                ...(raw.args !== undefined ? { data: raw.args } : {}),
              },
            },
            raw,
            session.session.threadId,
          );
          break;
        }

        case "tool_execution_update": {
          if (
            session.turnState === undefined ||
            typeof raw.toolCallId !== "string" ||
            typeof raw.toolName !== "string"
          ) {
            break;
          }
          const itemId = session.turnState.toolItemIdsByCallId.get(raw.toolCallId);
          if (!itemId) {
            break;
          }
          const detail =
            raw.partialResult !== undefined
              ? JSON.stringify(raw.partialResult).slice(0, 2_000)
              : undefined;
          yield* emit(
            {
              type: "item.updated",
              eventId: makeEventId(),
              provider: PROVIDER,
              threadId: session.session.threadId,
              turnId: session.turnState.turnId,
              itemId,
              createdAt,
              providerRefs: { providerItemId: ProviderItemId.make(raw.toolCallId) },
              payload: {
                itemType: toolItemType(raw.toolName),
                title: raw.toolName,
                ...(detail ? { detail } : {}),
              },
            },
            raw,
            session.session.threadId,
          );
          break;
        }

        case "tool_execution_end": {
          if (
            session.turnState === undefined ||
            typeof raw.toolCallId !== "string" ||
            typeof raw.toolName !== "string"
          ) {
            break;
          }
          const itemId =
            session.turnState.toolItemIdsByCallId.get(raw.toolCallId) ?? makeItemId(raw.toolName);
          const detail =
            raw.result !== undefined ? JSON.stringify(raw.result).slice(0, 4_000) : undefined;
          yield* emit(
            {
              type: "item.completed",
              eventId: makeEventId(),
              provider: PROVIDER,
              threadId: session.session.threadId,
              turnId: session.turnState.turnId,
              itemId,
              createdAt,
              providerRefs: { providerItemId: ProviderItemId.make(raw.toolCallId) },
              payload: {
                itemType: toolItemType(raw.toolName),
                title: raw.toolName,
                status: raw.isError === true ? "failed" : "completed",
                ...(detail ? { detail } : {}),
                ...(raw.result !== undefined ? { data: raw.result } : {}),
              },
            },
            raw,
            session.session.threadId,
          );
          break;
        }

        case "turn_end": {
          const turnState = session.turnState;
          if (!turnState) {
            break;
          }
          const message = raw.message as PiMessage | undefined;
          const usage = usageFromMessage(message);
          const assistantText = turnState.assistantText || textFromContent(message?.content);
          if (usage) {
            yield* emit(
              {
                type: "thread.token-usage.updated",
                eventId: makeEventId(),
                provider: PROVIDER,
                threadId: session.session.threadId,
                turnId: turnState.turnId,
                createdAt,
                payload: { usage },
              },
              raw,
              session.session.threadId,
            );
          }
          yield* emit(
            {
              type: "item.completed",
              eventId: makeEventId(),
              provider: PROVIDER,
              threadId: session.session.threadId,
              turnId: turnState.turnId,
              itemId: turnState.assistantItemId,
              createdAt,
              payload: {
                itemType: "assistant_message",
                ...(assistantText ? { detail: assistantText } : {}),
              },
            },
            raw,
            session.session.threadId,
          );
          yield* emit(
            {
              type: "turn.completed",
              eventId: makeEventId(),
              provider: PROVIDER,
              threadId: session.session.threadId,
              turnId: turnState.turnId,
              createdAt,
              payload: {
                state: turnStatusFromMessage(message),
                ...(message?.errorMessage ? { detail: message.errorMessage } : {}),
              },
            },
            raw,
            session.session.threadId,
          );
          yield* emit(
            {
              type: "session.state.changed",
              eventId: makeEventId(),
              provider: PROVIDER,
              threadId: session.session.threadId,
              createdAt,
              payload: { state: "ready" },
            },
            raw,
            session.session.threadId,
          );
          session.turns.push({ id: turnState.turnId, items: [...turnState.items] });
          session.turnState = undefined;
          break;
        }

        case "extension_ui_request": {
          yield* emit(
            {
              type: "runtime.warning",
              eventId: makeEventId(),
              provider: PROVIDER,
              threadId: session.session.threadId,
              createdAt,
              payload: {
                message:
                  "Pi requested extension UI input, which T3 Code does not currently support.",
              },
            },
            raw,
            session.session.threadId,
          );
          break;
        }

        default:
          break;
      }
    }).pipe(Effect.runPromise);

  const startSession: PiAdapterShape["startSession"] = (input) =>
    Effect.gen(function* () {
      if (input.provider !== undefined && input.provider !== PROVIDER) {
        return yield* toValidationError(
          "startSession",
          `Expected provider '${PROVIDER}' but received '${input.provider}'.`,
        );
      }
      const existing = sessions.get(input.threadId);
      if (existing) {
        return existing.session;
      }

      const settings: PiSettings = yield* getPiSettings(input.threadId, "startSession");
      const modelSelection =
        input.modelSelection?.provider === PROVIDER ? input.modelSelection : undefined;
      const initialModel = parsePiModelSlug(modelSelection?.model);
      if (modelSelection && !initialModel) {
        return yield* toValidationError(
          "startSession",
          "Pi models must use the `provider/model` format, for example `openai/gpt-5.4`.",
        );
      }

      const sessionFile =
        input.resumeCursor && typeof input.resumeCursor === "object" && input.resumeCursor !== null
          ? typeof (input.resumeCursor as { sessionFile?: unknown }).sessionFile === "string"
            ? (input.resumeCursor as { sessionFile: string }).sessionFile
            : sessionFilePath(input.threadId)
          : sessionFilePath(input.threadId);
      const cwd = input.cwd ?? config.cwd;

      const client = new PiRpcClient({
        binaryPath: settings.binaryPath,
        cwd,
        sessionFile,
        ...(initialModel ? { initialModel } : {}),
        onRecord: (raw) => {
          const current = sessions.get(input.threadId);
          if (current) {
            void handleRecord(current, raw);
          }
        },
        onExit: (exit) => {
          const current = sessions.get(input.threadId);
          if (current) {
            void handleUnexpectedExit(current, exit);
          }
        },
      });

      yield* Effect.tryPromise({
        try: () => client.start(),
        catch: (cause) =>
          new ProviderAdapterProcessError({
            provider: PROVIDER,
            threadId: input.threadId,
            detail: cause instanceof Error ? cause.message : "Failed to start Pi RPC process.",
            cause,
          }),
      });

      const state = yield* Effect.tryPromise({
        try: () => client.send<{ model?: { provider: string; id: string } }>({ type: "get_state" }),
        catch: (cause) =>
          new ProviderAdapterRequestError({
            provider: PROVIDER,
            method: "get_state",
            detail: cause instanceof Error ? cause.message : "Failed to query Pi session state.",
            cause,
          }),
      });

      const session: ProviderSession = {
        provider: PROVIDER,
        status: "ready",
        runtimeMode: input.runtimeMode,
        cwd,
        threadId: input.threadId,
        ...(state.model
          ? { model: toPiModelSlug({ provider: state.model.provider, modelId: state.model.id }) }
          : {}),
        resumeCursor: { sessionFile },
        createdAt: nowIso(),
        updatedAt: nowIso(),
      };

      const context: PiSessionContext = {
        session,
        client,
        sessionFile,
        turns: [],
        currentModel: session.model,
        turnState: undefined,
        nextTurnId: undefined,
        stopped: false,
      };
      sessions.set(input.threadId, context);
      yield* Effect.tryPromise({
        try: () => rebuildTurns(context),
        catch: (cause) =>
          new ProviderAdapterRequestError({
            provider: PROVIDER,
            method: "get_messages",
            detail: cause instanceof Error ? cause.message : "Failed to rebuild Pi thread history.",
            cause,
          }),
      }).pipe(Effect.ignore({ log: true }));

      const createdAt = nowIso();
      yield* emit({
        type: "session.started",
        eventId: makeEventId(),
        provider: PROVIDER,
        threadId: input.threadId,
        createdAt,
        payload: {
          resume: { sessionFile },
        },
      });
      yield* emit({
        type: "thread.started",
        eventId: makeEventId(),
        provider: PROVIDER,
        threadId: input.threadId,
        createdAt,
        payload: {},
      });
      yield* emit({
        type: "session.state.changed",
        eventId: makeEventId(),
        provider: PROVIDER,
        threadId: input.threadId,
        createdAt,
        payload: { state: "ready" },
      });

      return session;
    });

  const sendTurn: PiAdapterShape["sendTurn"] = (input) =>
    Effect.gen(function* () {
      const session = requireSession(input.threadId);
      const modelSelection =
        input.modelSelection?.provider === PROVIDER ? input.modelSelection : undefined;
      const requestedModel = parsePiModelSlug(modelSelection?.model);
      if (modelSelection && !requestedModel) {
        return yield* toValidationError(
          "sendTurn",
          "Pi models must use the `provider/model` format, for example `openai/gpt-5.4`.",
        );
      }

      if (requestedModel && session.currentModel !== modelSelection?.model) {
        yield* Effect.tryPromise({
          try: () =>
            session.client.send({
              type: "set_model",
              provider: requestedModel.provider,
              modelId: requestedModel.modelId,
            }),
          catch: (cause) =>
            new ProviderAdapterRequestError({
              provider: PROVIDER,
              method: "set_model",
              detail: cause instanceof Error ? cause.message : "Failed to switch Pi model.",
              cause,
            }),
        });
        session.currentModel = modelSelection?.model;
        session.session = {
          ...session.session,
          model: modelSelection?.model,
          updatedAt: nowIso(),
        };
      }

      const attachments = input.attachments ?? [];
      const images =
        attachments.length > 0
          ? yield* Effect.tryPromise({
              try: () =>
                Promise.all(
                  attachments.map((attachment) =>
                    readAttachmentAsPiImage({
                      attachmentsDir: config.attachmentsDir,
                      attachment,
                    }),
                  ),
                ),
              catch: (cause) =>
                new ProviderAdapterRequestError({
                  provider: PROVIDER,
                  method: "prompt",
                  detail:
                    cause instanceof Error ? cause.message : "Failed to load Pi image attachments.",
                  cause,
                }),
            })
          : [];

      const turnId = makeTurnId();
      session.nextTurnId = turnId;
      yield* Effect.tryPromise({
        try: () =>
          session.client.send({
            type: "prompt",
            message: input.input ?? "",
            ...(images.length > 0 ? { images } : {}),
          }),
        catch: (cause) =>
          new ProviderAdapterRequestError({
            provider: PROVIDER,
            method: "prompt",
            detail: cause instanceof Error ? cause.message : "Failed to send Pi prompt.",
            cause,
          }),
      });

      session.session = {
        ...session.session,
        status: "running",
        activeTurnId: turnId,
        updatedAt: nowIso(),
      };

      return {
        threadId: input.threadId,
        turnId,
        resumeCursor: { sessionFile: session.sessionFile },
      };
    });

  const interruptTurn: PiAdapterShape["interruptTurn"] = (threadId) =>
    Effect.tryPromise({
      try: async () => {
        const session = requireSession(threadId);
        await session.client.send({ type: "abort" });
      },
      catch: (cause) =>
        Schema.is(ProviderAdapterSessionNotFoundError)(cause)
          ? cause
          : new ProviderAdapterRequestError({
              provider: PROVIDER,
              method: "abort",
              detail: cause instanceof Error ? cause.message : "Failed to abort Pi turn.",
              cause,
            }),
    });

  const respondToRequest: PiAdapterShape["respondToRequest"] = () =>
    Effect.fail(
      new ProviderAdapterValidationError({
        provider: PROVIDER,
        operation: "respondToRequest",
        issue: "Pi does not expose interactive approval requests through this adapter.",
      }),
    );

  const respondToUserInput: PiAdapterShape["respondToUserInput"] = (
    _threadId,
    _requestId,
    _answers: ProviderUserInputAnswers,
  ) =>
    Effect.fail(
      new ProviderAdapterValidationError({
        provider: PROVIDER,
        operation: "respondToUserInput",
        issue: "Pi does not expose structured user-input requests through this adapter.",
      }),
    );

  const stopSession: PiAdapterShape["stopSession"] = (threadId) =>
    Effect.gen(function* () {
      const session = requireSession(threadId);
      session.stopped = true;
      yield* Effect.tryPromise({
        try: () => session.client.stop(),
        catch: (cause) =>
          new ProviderAdapterProcessError({
            provider: PROVIDER,
            threadId,
            detail: cause instanceof Error ? cause.message : "Failed to stop Pi process.",
            cause,
          }),
      });
      sessions.delete(threadId);
      const createdAt = nowIso();
      yield* emit({
        type: "session.exited",
        eventId: makeEventId(),
        provider: PROVIDER,
        threadId,
        createdAt,
        payload: {
          exitKind: "graceful",
        },
      });
    });

  const listSessions: PiAdapterShape["listSessions"] = () =>
    Effect.succeed(Array.from(sessions.values(), (session) => session.session));

  const hasSession: PiAdapterShape["hasSession"] = (threadId) =>
    Effect.succeed(sessions.has(threadId));

  const readThread: PiAdapterShape["readThread"] = (threadId) =>
    Effect.tryPromise({
      try: async () => {
        const session = requireSession(threadId);
        return rebuildTurns(session);
      },
      catch: (cause) =>
        Schema.is(ProviderAdapterSessionNotFoundError)(cause)
          ? cause
          : new ProviderAdapterRequestError({
              provider: PROVIDER,
              method: "get_messages",
              detail: cause instanceof Error ? cause.message : "Failed to read Pi thread.",
              cause,
            }),
    });

  const rollbackThread: PiAdapterShape["rollbackThread"] = (threadId, numTurns) =>
    Effect.tryPromise({
      try: async () => {
        const session = requireSession(threadId);
        if (numTurns <= 0) {
          return rebuildTurns(session);
        }
        const forkMessages = await session.client.send<{
          messages: Array<{ entryId: string; text: string }>;
        }>({ type: "get_fork_messages" });
        if (numTurns > forkMessages.messages.length) {
          throw new Error(
            `Cannot roll back ${numTurns} turns because Pi only has ${forkMessages.messages.length} user turns.`,
          );
        }
        const target = forkMessages.messages[forkMessages.messages.length - numTurns];
        if (!target) {
          return rebuildTurns(session);
        }
        await session.client.send({ type: "fork", entryId: target.entryId });
        const state = await session.client.send<{ model?: { provider: string; id: string } }>({
          type: "get_state",
        });
        session.currentModel = state.model
          ? toPiModelSlug({ provider: state.model.provider, modelId: state.model.id })
          : undefined;
        session.session = {
          ...session.session,
          ...(session.currentModel ? { model: session.currentModel } : {}),
          updatedAt: nowIso(),
        };
        return rebuildTurns(session);
      },
      catch: (cause) =>
        Schema.is(ProviderAdapterSessionNotFoundError)(cause)
          ? cause
          : Schema.is(ProviderAdapterValidationError)(cause)
            ? cause
            : new ProviderAdapterRequestError({
                provider: PROVIDER,
                method: "fork",
                detail: cause instanceof Error ? cause.message : "Failed to roll back Pi thread.",
                cause,
              }),
    });

  const stopAll: PiAdapterShape["stopAll"] = () =>
    Effect.forEach(
      Array.from(sessions.keys()),
      (threadId) => stopSession(threadId).pipe(Effect.ignore({ log: true })),
      { discard: true },
    );

  return {
    provider: PROVIDER,
    capabilities: { sessionModelSwitch: "in-session" },
    startSession,
    sendTurn,
    interruptTurn,
    respondToRequest,
    respondToUserInput,
    stopSession,
    listSessions,
    hasSession,
    readThread,
    rollbackThread,
    stopAll,
    streamEvents: Stream.fromQueue(runtimeEvents),
  } satisfies PiAdapterShape;
});

export const PiAdapterLive = Layer.effect(PiAdapter, makePiAdapter());

export function makePiAdapterLive(options?: PiAdapterLiveOptions) {
  return Layer.effect(PiAdapter, makePiAdapter(options));
}
