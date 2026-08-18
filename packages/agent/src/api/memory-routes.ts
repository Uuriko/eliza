/**
 * Memory + knowledge HTTP endpoints for the agent's dashboard/control API,
 * mounted behind that API's auth layer (returns 503 when no runtime is
 * attached).
 *
 * Hash-memory notes: POST /api/memory/remember stores a note, GET
 * /api/memory/search BM25-ranks them. GET /api/context/quick answers a query
 * over both hash-memory notes and documents via a TEXT_SMALL model call.
 * Memory viewer: GET /api/memories/feed | /browse | /by-entity/:id | /stats
 * read across the messages/memories/facts/documents tables; DELETE and PATCH
 * /api/memories/:id delete or edit-and-re-embed a single row (the id must look
 * like a UUID, keeping the literal sibling routes unambiguous).
 */
import crypto from "node:crypto";
import {
  type AgentRuntime,
  BM25,
  ChannelType,
  composePrompt,
  createMessageMemory,
  MESSAGE_SOURCE_CLIENT_CHAT,
  type Memory,
  ModelType,
  memoryContextQaTemplate,
  stringToUuid,
  type UUID,
} from "@elizaos/core";
import type { RouteRequestContext } from "@elizaos/shared";
import {
  PatchMemoryRequestSchema,
  PostMemoryRememberRequestSchema,
  parsePositiveInteger,
} from "@elizaos/shared";
import {
  type DocumentsServiceResult,
  getDocumentsService,
} from "./documents-service-loader.ts";
import { decodePathComponent } from "./server-helpers.ts";

export const HASH_MEMORY_SOURCE = "hash_memory";
const UUID_REGEX =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const MEMORY_SEARCH_SCAN_LIMIT = 2_000;
const MEMORY_SEARCH_DEFAULT_LIMIT = 10;
const MEMORY_SEARCH_MAX_LIMIT = 50;
const QUICK_CONTEXT_DEFAULT_LIMIT = 8;
const QUICK_CONTEXT_MAX_LIMIT = 20;
const QUICK_CONTEXT_DOCUMENTS_THRESHOLD = 0.2;

const MEMORY_BROWSE_DEFAULT_LIMIT = 50;
const MEMORY_BROWSE_MAX_LIMIT = 200;
/** Smallest per-table adapter page for the browse/feed drain. */
const MEMORY_BROWSE_MIN_PAGE = 200;
/**
 * Extra rows a table may be scanned past the requested page before the drain
 * stops and the response marks its total incomplete. Bounds the in-process
 * work of a filter the adapter cannot apply (entity, empty text, multi-term
 * keyword) without ever capping an unfiltered deep offset.
 */
const MEMORY_BROWSE_SCAN_SLACK = 10_000;
const MEMORY_FEED_DEFAULT_LIMIT = 50;
const MEMORY_FEED_MAX_LIMIT = 100;
const MEMORY_TABLE_NAMES = [
  "messages",
  "memories",
  "facts",
  "documents",
] as const;

export interface MemoryRouteContext extends RouteRequestContext {
  url: URL;
  runtime: AgentRuntime | null;
  agentName: string;
}

type MemorySearchHit = {
  id: string;
  text: string;
  createdAt: number;
  score: number;
};

type DocumentSearchHit = {
  id: string;
  text: string;
  similarity: number;
  documentId?: string;
  documentTitle?: string;
  position?: number;
};

type DocumentSearchMatch = {
  id: UUID;
  content: { text?: string };
  similarity?: number;
  metadata?: Record<string, unknown>;
};

function resolveAgentName(runtime: AgentRuntime, fallbackName: string): string {
  return runtime.character.name?.trim() || fallbackName || "Eliza";
}

async function ensureMemoryConnection(
  runtime: AgentRuntime,
  agentName: string,
): Promise<{ roomId: UUID; entityId: UUID }> {
  const entityId = runtime.agentId as UUID;
  const roomId = stringToUuid(`${agentName}-hash-memory-room`) as UUID;
  const worldId = stringToUuid(`${agentName}-hash-memory-world`) as UUID;
  const messageServerId = stringToUuid(
    `${agentName}-hash-memory-server`,
  ) as UUID;

  await runtime.ensureConnection({
    entityId,
    roomId,
    worldId,
    userName: "User",
    source: MESSAGE_SOURCE_CLIENT_CHAT,
    channelId: `${agentName}-hash-memory`,
    type: ChannelType.DM,
    messageServerId,
    metadata: { ownership: { ownerId: entityId } },
  });

  return { roomId, entityId };
}

/**
 * Rank a candidate set against `query` with Okapi BM25 + Porter2 stemming,
 * returning each item with a [0,1] max-normalized relevance score in input order.
 *
 * Corpus-aware IDF down-weights filler/stop words and TF saturation + length
 * normalization rank genuinely-relevant text first; a naive pairwise substring
 * count with no IDF would let a doc that merely contains a common query word
 * ("the") tie with a real hit. We use the `search.ts` BM25
 * (not the documents `bm25Scores`) specifically for its **Porter2 stemming** —
 * short typed chat queries are usually base forms ("configure") while stored
 * messages carry inflected forms ("configuring"/"configured"/"configuration"),
 * and stemming is the standard keyword answer to that mismatch. It also brings
 * stop-word removal and proper Unicode/accent/CJK normalization (the documents
 * tokenizer strips non-ASCII, silently dropping accented + non-Latin text).
 */
export function rankByKeyword<T>(
  query: string,
  items: T[],
  getText: (item: T) => string,
): Array<{ item: T; score: number }> {
  if (items.length === 0) return [];
  // Single `content` field per doc so only the text is indexed; items are
  // tracked by array index (the BM25 result `index`).
  const bm25 = new BM25(
    items.map((item) => ({ content: getText(item) })),
    { stemming: true },
  );
  const results = bm25.search(query, items.length);
  if (results.length === 0) return items.map((item) => ({ item, score: 0 }));
  const firstScore = results[0]?.score;
  const maxScore = typeof firstScore === "number" ? firstScore : 0;
  const scoreByIndex = new Map(results.map((r) => [r.index, r.score]));
  return items.map((item, i) => {
    const indexedScore = scoreByIndex.get(i);
    return {
      item,
      score:
        maxScore > 0
          ? (typeof indexedScore === "number" ? indexedScore : 0) / maxScore
          : 0,
    };
  });
}

/**
 * Boolean keyword match for *filtering* (not ranking): does the text contain the
 * whole query or any query term (≥2 chars)? Used where the caller wants
 * "messages matching this text", not a relevance ranking.
 */
export function matchesKeyword(text: string, query: string): boolean {
  const normalizedText = text.toLowerCase();
  const normalizedQuery = query.toLowerCase().trim();
  if (!normalizedText || !normalizedQuery) return false;
  if (normalizedText.includes(normalizedQuery)) return true;
  return normalizedQuery
    .split(/\s+/)
    .filter((term) => term.length >= 2)
    .some((term) => normalizedText.includes(term));
}

async function searchMemoryNotes(
  runtime: AgentRuntime,
  roomId: UUID,
  query: string,
  limit: number,
): Promise<MemorySearchHit[]> {
  const memories = await runtime.getMemories({
    roomId,
    tableName: "messages",
    limit: MEMORY_SEARCH_SCAN_LIMIT,
    includeEmbedding: false, // only reads content.text
  });

  // Gather hash-memory candidates first, then BM25-rank them as a corpus.
  const candidates: Array<{ id: UUID; text: string; createdAt: number }> = [];
  for (const memory of memories) {
    const text = (
      memory.content as { text?: string } | undefined
    )?.text?.trim();
    if (!text) continue;
    const source = (memory.content as { source?: string } | undefined)?.source;
    if (source !== HASH_MEMORY_SOURCE) continue;
    if (!memory.id || typeof memory.createdAt !== "number") continue;
    candidates.push({
      id: memory.id,
      text,
      createdAt: memory.createdAt,
    });
  }

  const hits: MemorySearchHit[] = rankByKeyword(
    query,
    candidates,
    (c) => c.text,
  )
    .filter(({ score }) => score > 0)
    .map(({ item, score }) => ({
      id: item.id,
      text: item.text,
      createdAt: item.createdAt,
      score,
    }));

  hits.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return b.createdAt - a.createdAt;
  });
  return hits.slice(0, limit);
}

async function searchDocuments(
  runtime: AgentRuntime,
  query: string,
  limit: number,
): Promise<DocumentSearchHit[]> {
  const documents: DocumentsServiceResult = await getDocumentsService(runtime);
  const documentsService = documents.service;
  if (!documentsService || !runtime.agentId) return [];

  const agentId = runtime.agentId as UUID;
  const searchMessage: Memory = {
    id: crypto.randomUUID() as UUID,
    entityId: agentId,
    agentId,
    roomId: agentId,
    content: { text: query },
    createdAt: Date.now(),
  };

  const matches: DocumentSearchMatch[] = await documentsService.searchDocuments(
    searchMessage,
    {
      roomId: agentId,
    },
  );

  return matches
    .filter(
      (match) => (match.similarity ?? 0) >= QUICK_CONTEXT_DOCUMENTS_THRESHOLD,
    )
    .slice(0, limit)
    .map((match) => {
      const metadata = match.metadata as Record<string, unknown> | undefined;
      return {
        id: match.id,
        text: match.content.text ?? "",
        similarity: match.similarity ?? 0,
        documentId:
          typeof metadata?.documentId === "string"
            ? metadata.documentId
            : undefined,
        documentTitle:
          typeof metadata?.filename === "string"
            ? metadata.filename
            : typeof metadata?.title === "string"
              ? metadata.title
              : undefined,
        position:
          typeof metadata?.position === "number"
            ? metadata.position
            : undefined,
      };
    });
}

function buildQuickContextPrompt(params: {
  query: string;
  memories: MemorySearchHit[];
  documents: DocumentSearchHit[];
}): string {
  const { query, memories, documents } = params;
  const memorySection =
    memories.length > 0
      ? memories
          .map((item, index) => `- [M${index + 1}] ${item.text}`)
          .join("\n")
      : "- none";
  const documentsSection =
    documents.length > 0
      ? documents
          .map((item, index) => `- [D${index + 1}] ${item.text}`)
          .join("\n")
      : "- none";

  return composePrompt({
    state: { query, memorySection, knowledgeSection: documentsSection },
    template: memoryContextQaTemplate,
  });
}

type MemoryBrowseItem = {
  id: string;
  type: string;
  text: string;
  entityId: string | null;
  roomId: string | null;
  agentId: string | null;
  createdAt: number;
  metadata: Record<string, unknown> | null;
  source: string | null;
};

type TaggedMemory = Memory & { _table: string };

/** Ordering key — `Memory.createdAt` is optional; rows without one sort as oldest. */
const memoryCreatedAt = (memory: { createdAt?: number }): number =>
  memory.createdAt ?? 0;

/** Newest-first comparator shared by the browse/search/feed list routes. */
const byNewestFirst = (
  a: { createdAt?: number },
  b: { createdAt?: number },
): number => memoryCreatedAt(b) - memoryCreatedAt(a);

function memoryToBrowseItem(memory: TaggedMemory): MemoryBrowseItem {
  const content = memory.content as Record<string, unknown> | undefined;
  return {
    id: memory.id ?? "",
    type: memory._table,
    text: (content?.text as string) ?? "",
    entityId: memory.entityId,
    roomId: memory.roomId,
    agentId: memory.agentId ?? null,
    createdAt: memoryCreatedAt(memory),
    metadata: (memory.metadata as Record<string, unknown>) ?? null,
    source: (content?.source as string) ?? null,
  };
}

function hasBrowsableContent(memory: TaggedMemory): boolean {
  const text = (memory.content as { text?: string } | undefined)?.text;
  return typeof text === "string" && text.trim().length > 0;
}

/**
 * Read newest-first rows from each table, in growing offset windows, until the
 * table has yielded `need` rows that pass every filter or has no rows left.
 * Filters the adapter supports (`roomId`, `end`, `textContains`) are pushed
 * down; the rest (`entityIds` — the adapter treats `entityId` as RLS context
 * only — empty text, `before`, `accept`) are applied here, which is why the
 * scan must keep going instead of trusting a fixed over-fetch window (#22061).
 *
 * Each table's newest `need` kept rows are enough for any newest-first page
 * ending at `need`, so the caller may merge, sort and slice. `exhausted` is
 * true only when every table ran dry, in which case the returned count is
 * exact. A table that neither ran dry nor reached `need` within
 * `MEMORY_BROWSE_SCAN_SLACK` extra rows stops early; callers must then report
 * the count as incomplete rather than exact.
 */
async function drainMemoriesFromTables(
  runtime: AgentRuntime,
  params: {
    entityIds?: UUID[];
    roomId?: UUID;
    tables?: readonly string[];
    before?: number;
    textContains?: string;
    accept?: (memory: TaggedMemory) => boolean;
    need: number;
  },
): Promise<{ memories: TaggedMemory[]; exhausted: boolean }> {
  const tables = params.tables ?? MEMORY_TABLE_NAMES;
  const entityIds = params.entityIds?.length
    ? new Set<string>(params.entityIds)
    : null;
  const before = params.before;
  const scanCap = params.need + MEMORY_BROWSE_SCAN_SLACK;
  const keep = (m: TaggedMemory): boolean =>
    hasBrowsableContent(m) &&
    (entityIds === null || (!!m.entityId && entityIds.has(m.entityId))) &&
    (before === undefined || memoryCreatedAt(m) < before) &&
    (params.accept === undefined || params.accept(m));

  // Tables are independent queries; read them concurrently. Promise.all
  // preserves input order, so the flattened result is order-stable.
  const perTable = await Promise.all(
    tables.map(async (tableName) => {
      const kept: TaggedMemory[] = [];
      let scanned = 0;
      let pageLimit = Math.max(params.need, MEMORY_BROWSE_MIN_PAGE);
      while (kept.length < params.need) {
        const limit = Math.min(pageLimit, scanCap - scanned);
        if (limit <= 0) return { kept, exhausted: false };
        const rows = await runtime.getMemories({
          agentId: runtime.agentId as UUID,
          roomId: params.roomId,
          tableName,
          limit,
          offset: scanned,
          // `end` is inclusive; `before` is exclusive at millisecond precision.
          end: before === undefined ? undefined : before - 1,
          textContains: params.textContains,
          includeEmbedding: false, // browse feed discards embeddings (memoryToBrowseItem)
        });
        scanned += rows.length;
        for (const row of rows) {
          const tagged = Object.assign(row, { _table: tableName });
          if (keep(tagged)) kept.push(tagged);
        }
        if (rows.length < limit) return { kept, exhausted: true };
        // Double the window (never re-reading the prefix) so a sparse filter
        // converges in O(log n) round-trips instead of O(n) fixed pages.
        pageLimit *= 2;
      }
      return { kept, exhausted: false };
    }),
  );
  return {
    memories: perTable.flatMap((t) => t.kept),
    exhausted: perTable.every((t) => t.exhausted),
  };
}

/**
 * Parse the memory-viewer `type` query. Omitted/empty means every table
 * (the "all" tab). A known table name narrows the scan. Any other token
 * used to fall through to that same unfiltered scan, so `type=notes` or
 * `type=message` silently returned the whole feed.
 */
export function parseMemoryTableFilter(
  typeParam: string | null,
): { ok: true; tables?: readonly string[] } | { ok: false; message: string } {
  if (typeParam === null || typeParam === "") return { ok: true };
  const t = typeParam.toLowerCase();
  if (MEMORY_TABLE_NAMES.includes(t as (typeof MEMORY_TABLE_NAMES)[number])) {
    return { ok: true, tables: [t] };
  }
  return {
    ok: false,
    message: `type must be one of: ${MEMORY_TABLE_NAMES.join(", ")}`,
  };
}

export async function handleMemoryRoutes(
  ctx: MemoryRouteContext,
): Promise<boolean> {
  const {
    req,
    res,
    method,
    pathname,
    url,
    runtime,
    agentName,
    json,
    error,
    readJsonBody,
  } = ctx;

  if (
    !pathname.startsWith("/api/memory") &&
    !pathname.startsWith("/api/memories") &&
    pathname !== "/api/context/quick"
  ) {
    return false;
  }

  if (!runtime) {
    error(res, "Agent runtime is not available", 503);
    return true;
  }

  const resolvedAgentName = resolveAgentName(runtime, agentName);
  const { roomId, entityId } = await ensureMemoryConnection(
    runtime,
    resolvedAgentName,
  );

  if (method === "POST" && pathname === "/api/memory/remember") {
    const rawRem = await readJsonBody<Record<string, unknown>>(req, res);
    if (rawRem === null) return true;
    const parsedRem = PostMemoryRememberRequestSchema.safeParse(rawRem);
    if (!parsedRem.success) {
      error(res, parsedRem.error.issues[0]?.message ?? "text is required", 400);
      return true;
    }
    const text = parsedRem.data.text;
    const createdAt = Date.now();
    const memoryId = parsedRem.data.idempotencyKey
      ? (stringToUuid(
          `${HASH_MEMORY_SOURCE}:${runtime.agentId}:${parsedRem.data.idempotencyKey}`,
        ) as UUID)
      : (crypto.randomUUID() as UUID);
    const existing = parsedRem.data.idempotencyKey
      ? await runtime.getMemoryById(memoryId)
      : null;
    if (existing) {
      json(res, {
        ok: true,
        id: existing.id,
        text: existing.content.text ?? text,
        createdAt: existing.createdAt,
        replayed: true,
      });
      return true;
    }
    const message = createMessageMemory({
      id: memoryId,
      entityId,
      roomId,
      content: {
        text,
        source: HASH_MEMORY_SOURCE,
        channelType: ChannelType.DM,
      },
    });
    await runtime.createMemory(message, "messages");
    json(res, {
      ok: true,
      id: message.id,
      text,
      createdAt,
      replayed: false,
    });
    return true;
  }

  if (method === "GET" && pathname === "/api/memory/search") {
    const query = url.searchParams.get("q")?.trim() ?? "";
    if (!query) {
      error(res, "Search query (q) is required", 400);
      return true;
    }
    const requestedLimit = parsePositiveInteger(
      url.searchParams.get("limit"),
      MEMORY_SEARCH_DEFAULT_LIMIT,
    );
    const limit = Math.min(
      Math.max(requestedLimit, 1),
      MEMORY_SEARCH_MAX_LIMIT,
    );
    const results = await searchMemoryNotes(runtime, roomId, query, limit);
    json(res, {
      query,
      results,
      count: results.length,
      limit,
    });
    return true;
  }

  if (method === "GET" && pathname === "/api/context/quick") {
    const query = url.searchParams.get("q")?.trim() ?? "";
    if (!query) {
      error(res, "Search query (q) is required", 400);
      return true;
    }
    const requestedLimit = parsePositiveInteger(
      url.searchParams.get("limit"),
      QUICK_CONTEXT_DEFAULT_LIMIT,
    );
    const limit = Math.min(
      Math.max(requestedLimit, 1),
      QUICK_CONTEXT_MAX_LIMIT,
    );

    const [memories, documents] = await Promise.all([
      searchMemoryNotes(runtime, roomId, query, limit),
      searchDocuments(runtime, query, limit),
    ]);

    const prompt = buildQuickContextPrompt({ query, memories, documents });
    let answer = "I couldn't generate a quick answer right now.";
    const response = await runtime.useModel(ModelType.TEXT_SMALL, { prompt });
    const text = typeof response === "string" ? response : String(response);
    if (text.trim()) {
      answer = text.trim();
    }

    json(res, {
      query,
      answer,
      memories,
      documents,
    });
    return true;
  }

  // ── Memory Viewer endpoints ───────────────────────────────────────────

  if (method === "GET" && pathname === "/api/memories/feed") {
    const requestedLimit = parsePositiveInteger(
      url.searchParams.get("limit"),
      MEMORY_FEED_DEFAULT_LIMIT,
    );
    const limit = Math.min(Math.max(requestedLimit, 1), MEMORY_FEED_MAX_LIMIT);
    const beforeParam = url.searchParams.get("before");
    const before = beforeParam ? Number(beforeParam) : undefined;
    // Number("abc") is NaN, which is a `number` and so survives the
    // `beforeTs !== undefined` cursor gate below — but every `< NaN`
    // comparison is false, so a malformed cursor would silently empty the
    // feed with a 200. Reject it here instead, mirroring the `type` guard.
    if (before !== undefined && !Number.isFinite(before)) {
      error(res, "before must be a finite Unix timestamp in milliseconds", 400);
      return true;
    }
    const tableFilter = parseMemoryTableFilter(url.searchParams.get("type"));
    if (!tableFilter.ok) {
      error(res, tableFilter.message, 400);
      return true;
    }
    const tables = tableFilter.tables;

    const { memories: allMemories, exhausted } = await drainMemoriesFromTables(
      runtime,
      { tables, before, need: limit + 1 },
    );

    allMemories.sort(byNewestFirst);
    const items = allMemories.slice(0, limit).map(memoryToBrowseItem);

    json(res, {
      memories: items,
      count: items.length,
      limit,
      // Honest even when the drain stopped short: rows we never looked at
      // may still be browsable, so only a dry table set can say "no more".
      hasMore: allMemories.length > limit || !exhausted,
    });
    return true;
  }

  if (method === "GET" && pathname === "/api/memories/browse") {
    const requestedLimit = parsePositiveInteger(
      url.searchParams.get("limit"),
      MEMORY_BROWSE_DEFAULT_LIMIT,
    );
    const limit = Math.min(
      Math.max(requestedLimit, 1),
      MEMORY_BROWSE_MAX_LIMIT,
    );
    const offset = parsePositiveInteger(url.searchParams.get("offset"), 0);
    const tableFilter = parseMemoryTableFilter(url.searchParams.get("type"));
    if (!tableFilter.ok) {
      error(res, tableFilter.message, 400);
      return true;
    }
    const tables = tableFilter.tables;
    const entityIdParam = url.searchParams.get("entityId");
    const entityIdsParam = url.searchParams.get("entityIds");
    const roomIdParam = url.searchParams.get("roomId");
    const searchQuery = url.searchParams.get("q")?.trim() ?? "";

    const entityIds: UUID[] | undefined = entityIdsParam
      ? (entityIdsParam
          .split(",")
          .map((id) => id.trim())
          .filter(Boolean) as UUID[])
      : entityIdParam
        ? [entityIdParam as UUID]
        : undefined;

    // A single-term query is exactly the adapter's case-insensitive substring
    // predicate, so push it down; multi-term queries keep matchesKeyword's
    // any-term semantics and are filtered here (matchesKeyword still runs on
    // every row either way).
    const textContains =
      searchQuery && !/\s/.test(searchQuery) ? searchQuery : undefined;
    const { memories: filtered, exhausted } = await drainMemoriesFromTables(
      runtime,
      {
        tables,
        entityIds,
        roomId: roomIdParam ? (roomIdParam as UUID) : undefined,
        textContains,
        accept: searchQuery
          ? (m) =>
              matchesKeyword(
                (m.content as { text?: string } | undefined)?.text ?? "",
                searchQuery,
              )
          : undefined,
        need: offset + limit + 1,
      },
    );

    filtered.sort(byNewestFirst);
    const page = filtered.slice(offset, offset + limit).map(memoryToBrowseItem);

    json(res, {
      memories: page,
      // Exact when every table was read to the end; otherwise a lower bound
      // that still exceeds offset+limit whenever this page was filled.
      total: filtered.length,
      totalIsExact: exhausted,
      limit,
      offset,
    });
    return true;
  }

  if (method === "GET" && pathname.startsWith("/api/memories/by-entity/")) {
    const primaryEntityId = decodePathComponent(
      pathname.slice("/api/memories/by-entity/".length),
      res,
      "entity identifier",
    );
    if (primaryEntityId === null) return true;
    if (!UUID_REGEX.test(primaryEntityId)) {
      error(res, "Invalid entity identifier.", 400);
      return true;
    }

    // Support multi-identity people: ?entityIds=id1,id2,id3
    // Falls back to the single path param if not provided.
    const entityIdsParam = url.searchParams.get("entityIds");
    const entityIds: UUID[] = entityIdsParam
      ? (entityIdsParam
          .split(",")
          .map((id) => id.trim())
          .filter(Boolean) as UUID[])
      : [primaryEntityId as UUID];

    const requestedLimit = parsePositiveInteger(
      url.searchParams.get("limit"),
      MEMORY_BROWSE_DEFAULT_LIMIT,
    );
    const limit = Math.min(
      Math.max(requestedLimit, 1),
      MEMORY_BROWSE_MAX_LIMIT,
    );
    const offset = parsePositiveInteger(url.searchParams.get("offset"), 0);
    const tableFilter = parseMemoryTableFilter(url.searchParams.get("type"));
    if (!tableFilter.ok) {
      error(res, tableFilter.message, 400);
      return true;
    }
    const tables = tableFilter.tables;

    const { memories: allMemories, exhausted } = await drainMemoriesFromTables(
      runtime,
      { entityIds, tables, need: offset + limit + 1 },
    );

    allMemories.sort(byNewestFirst);
    const page = allMemories
      .slice(offset, offset + limit)
      .map(memoryToBrowseItem);

    json(res, {
      entityId: primaryEntityId,
      memories: page,
      total: allMemories.length,
      totalIsExact: exhausted,
      limit,
      offset,
    });
    return true;
  }

  // ── Memory mutation by id ─────────────────────────────────────────────
  // DELETE /api/memories/:id and PATCH /api/memories/:id operate on the bare
  // id segment. Path matching only fires when the segment looks like a UUID,
  // which keeps the literal sibling routes (`feed`, `browse`, `stats`,
  // `by-entity/...`) unambiguous.

  const memoryIdMatch = /^\/api\/memories\/([^/]+)$/.exec(pathname);
  if (memoryIdMatch && (method === "DELETE" || method === "PATCH")) {
    const rawId = decodePathComponent(memoryIdMatch[1] ?? "", res, "memory id");
    if (rawId === null) return true;
    if (!UUID_REGEX.test(rawId)) {
      error(res, "Invalid memory id.", 400);
      return true;
    }
    const memoryId = rawId as UUID;
    const existing = await runtime.getMemoryById(memoryId);
    if (!existing) {
      error(res, "Memory not found.", 404);
      return true;
    }

    if (method === "DELETE") {
      await runtime.deleteMemory(memoryId);
      json(res, { deleted: true, id: memoryId });
      return true;
    }

    // PATCH — update text, regenerate embedding, then atomically persist
    // both via runtime.updateMemory (the SQL adapter writes content +
    // embedding in a single transaction). If embedding generation fails we
    // return 500 *before* touching the database, so there is nothing to roll
    // back.
    const rawPat = await readJsonBody<Record<string, unknown>>(req, res);
    if (rawPat === null) return true;
    const parsedPat = PatchMemoryRequestSchema.safeParse(rawPat);
    if (!parsedPat.success) {
      error(res, parsedPat.error.issues[0]?.message ?? "text is required", 400);
      return true;
    }
    const text = parsedPat.data.text;

    const existingContent =
      (existing.content as Record<string, unknown> | undefined) ?? {};
    const nextContent = { ...existingContent, text };

    let embedding: number[];
    try {
      embedding = await runtime.useModel(ModelType.TEXT_EMBEDDING, {
        text,
      });
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      error(res, `Failed to regenerate embedding: ${detail}`, 500);
      return true;
    }
    if (!Array.isArray(embedding) || embedding.length === 0) {
      error(res, "Embedding model returned no vector.", 500);
      return true;
    }

    await runtime.updateMemory({
      id: memoryId,
      content: nextContent,
      embedding,
    });

    const updated = await runtime.getMemoryById(memoryId);
    json(res, { updated: true, id: memoryId, memory: updated });
    return true;
  }

  if (method === "GET" && pathname === "/api/memories/stats") {
    const counts: Record<string, number> = {};
    let total = 0;

    for (const tableName of MEMORY_TABLE_NAMES) {
      const memories = await runtime.getMemories({
        agentId: runtime.agentId as UUID,
        tableName,
        limit: 10000,
        includeEmbedding: false, // stats only counts memories.length
      });
      counts[tableName] = memories.length;
      total += memories.length;
    }

    json(res, { total, byType: counts });
    return true;
  }

  return false;
}
