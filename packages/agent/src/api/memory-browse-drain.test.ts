/**
 * Exercises the memory browse/by-entity/feed drain at the real
 * handleMemoryRoutes boundary with a deterministic fake adapter that honors
 * `limit`, `offset`, `end` and `textContains` the way plugin-sql does
 * (newest-first). Covers the three #22061 defects — sparse keyword filters
 * losing deep matches with a lying total, by-entity post-filter truncation,
 * and the feed's empty-text hasMore false-negative — plus the #22066 close
 * contract: unfiltered deep offsets are never capped, incomplete totals are
 * observable, and the drain never re-reads a prefix.
 */

import type { AgentRuntime, Memory, UUID } from "@elizaos/core";
import { describe, expect, test, vi } from "vitest";
import type { MemoryRouteContext } from "./memory-routes.ts";
import { handleMemoryRoutes } from "./memory-routes.ts";

const AGENT_ID = "11111111-1111-4111-8111-111111111111" as UUID;
const ENTITY = "22222222-2222-4222-8222-222222222222" as UUID;
const OTHER = "33333333-3333-4333-8333-333333333333" as UUID;
const ROOM = "44444444-4444-4444-8444-444444444444" as UUID;

function makeRow(i: number, text: string, entityId: UUID = OTHER): Memory {
  return {
    id: `00000000-0000-4000-8000-${String(i).padStart(12, "0")}` as UUID,
    entityId,
    roomId: ROOM,
    agentId: AGENT_ID,
    createdAt: 2_000_000 - i, // strictly newest-first
    content: { text },
  } as Memory;
}

/** 1000 messages: every 10th contains "needle" (100), every 20th owned by ENTITY (50). */
function buildDb(): Memory[] {
  const db: Memory[] = [];
  for (let i = 0; i < 1000; i++) {
    db.push(
      makeRow(
        i,
        i % 10 === 0 ? `needle row ${i}` : `hay row ${i}`,
        i % 20 === 0 ? ENTITY : OTHER,
      ),
    );
  }
  return db;
}

type GetMemoriesParams = {
  tableName: string;
  limit: number;
  offset?: number;
  end?: number;
  textContains?: string;
};

/**
 * Fake adapter over a newest-first `messages` table: applies `end` and
 * `textContains` like plugin-sql (ignores entityId, which is RLS-only there),
 * then windows by offset/limit.
 */
function makeRuntime(
  db: Memory[],
  options: { honorTextContains?: boolean } = {},
): {
  runtime: AgentRuntime;
  getMemories: ReturnType<typeof vi.fn>;
} {
  const honorText = options.honorTextContains ?? true;
  const getMemories = vi.fn(async (params: GetMemoriesParams) => {
    if (params.tableName !== "messages") return [];
    let view = db;
    if (params.end !== undefined) {
      const end = params.end;
      view = view.filter((r) => (r.createdAt ?? 0) <= end);
    }
    if (honorText && params.textContains) {
      const needle = params.textContains.toLowerCase();
      view = view.filter((r) =>
        String(r.content.text ?? "")
          .toLowerCase()
          .includes(needle),
      );
    }
    const offset = params.offset ?? 0;
    const page = view
      .slice(offset, offset + params.limit)
      .map((r) => ({ ...r }));
    return page;
  });
  const runtime = {
    agentId: AGENT_ID,
    character: { name: "Eliza" },
    ensureConnection: vi.fn(async () => undefined),
    getMemories,
  } as unknown as AgentRuntime;
  return { runtime, getMemories };
}

async function get(
  runtime: AgentRuntime,
  path: string,
): Promise<Record<string, unknown>> {
  const url = new URL(`https://agent.test${path}`);
  let out: unknown;
  const context: MemoryRouteContext = {
    req: {} as never,
    res: {} as never,
    method: "GET",
    pathname: url.pathname,
    url,
    runtime,
    agentName: "Eliza",
    json: (_res, value) => {
      out = value;
    },
    error: (_res, message, status) => {
      throw new Error(`unexpected ${status}: ${message}`);
    },
    readJsonBody: async <T extends object>() => ({}) as T,
  };
  expect(await handleMemoryRoutes(context)).toBe(true);
  return out as Record<string, unknown>;
}

const ids = (res: Record<string, unknown>): string[] =>
  (res.memories as Array<{ id: string }>).map((m) => m.id);

describe("memory browse drain (#22061)", () => {
  test("sparse q reaches every match across pages; total never lies about exhaustion", async () => {
    // Adapter that ignores textContains forces the in-process drain path.
    const { runtime } = makeRuntime(buildDb(), { honorTextContains: false });
    const union = new Set<string>();
    let offset = 0;
    let page: Record<string, unknown>;
    for (;;) {
      page = await get(
        runtime,
        `/api/memories/browse?type=messages&q=needle&limit=50&offset=${offset}`,
      );
      for (const id of ids(page)) union.add(id);
      // While matches remain, total must exceed the page end (Next enabled)
      // and be flagged incomplete unless the table was fully read.
      if (offset + 50 >= (page.total as number)) break;
      expect(ids(page)).toHaveLength(50);
      offset += 50;
    }
    expect(union.size).toBe(100);
    expect(page.total).toBe(100);
    expect(page.totalIsExact).toBe(true);
  });

  test("first page: total is a lower bound past the window and marked incomplete", async () => {
    const { runtime } = makeRuntime(buildDb(), { honorTextContains: false });
    const p0 = await get(
      runtime,
      "/api/memories/browse?type=messages&q=needle&limit=50&offset=0",
    );
    expect(ids(p0)).toHaveLength(50);
    expect(p0.total as number).toBeGreaterThan(50);
    expect(p0.totalIsExact).toBe(false);
  });

  test("single-term q is pushed down as textContains: one adapter call, exact total", async () => {
    const { runtime, getMemories } = makeRuntime(buildDb());
    const res = await get(
      runtime,
      "/api/memories/browse?type=messages&q=needle&limit=50&offset=0",
    );
    expect(ids(res)).toHaveLength(50);
    expect(res.total).toBe(100);
    expect(res.totalIsExact).toBe(true);
    expect(getMemories).toHaveBeenCalledTimes(1);
    expect(getMemories.mock.calls[0][0]).toMatchObject({
      textContains: "needle",
      offset: 0,
    });
  });

  test("by-entity drains past the adapter's entityId blindness: all 50 rows, exact total", async () => {
    const { runtime } = makeRuntime(buildDb());
    const res = await get(
      runtime,
      `/api/memories/by-entity/${ENTITY}?type=messages&limit=50&offset=0`,
    );
    expect(ids(res)).toHaveLength(50);
    expect(res.total).toBe(50);
    expect(res.totalIsExact).toBe(true);
    expect(
      (res.memories as Array<{ entityId: string }>).every(
        (m) => m.entityId === ENTITY,
      ),
    ).toBe(true);
  });

  test("feed hasMore is true when browsable rows exist past an empty-text run", async () => {
    // Newest 30 rows with text, next 870 empty, oldest 100 with text.
    const db: Memory[] = [];
    for (let i = 0; i < 1000; i++) {
      db.push(makeRow(i, i < 30 || i >= 900 ? `txt ${i}` : ""));
    }
    const { runtime } = makeRuntime(db);
    const res = await get(runtime, "/api/memories/feed?type=messages&limit=50");
    expect(res.count).toBe(50);
    expect(res.hasMore).toBe(true);
    // Second page via the cursor the UI sends (oldest returned createdAt).
    const oldest = (res.memories as Array<{ createdAt: number }>).at(-1)
      ?.createdAt as number;
    const next = await get(
      runtime,
      `/api/memories/feed?type=messages&limit=50&before=${oldest}`,
    );
    expect(next.count).toBe(50);
    expect(next.hasMore).toBe(true);
    const last = (next.memories as Array<{ createdAt: number }>).at(-1)
      ?.createdAt as number;
    const tail = await get(
      runtime,
      `/api/memories/feed?type=messages&limit=50&before=${last}`,
    );
    expect(tail.count).toBe(30);
    expect(tail.hasMore).toBe(false);
  });

  test("unfiltered deep offset develop serves today is still served, in one call", async () => {
    const db: Memory[] = [];
    for (let i = 0; i < 25_000; i++) db.push(makeRow(i, `row ${i}`));
    const { runtime, getMemories } = makeRuntime(db);
    const res = await get(
      runtime,
      "/api/memories/browse?type=messages&limit=50&offset=20000",
    );
    expect(ids(res)).toHaveLength(50);
    expect(ids(res)[0]).toBe(makeRow(20_000, "").id);
    expect(res.total as number).toBeGreaterThan(20_050);
    expect(res.totalIsExact).toBe(false);
    expect(getMemories).toHaveBeenCalledTimes(1);
  });

  test("a lone match below the first window but within the scan slack is found", async () => {
    const db: Memory[] = [];
    for (let i = 0; i < 5000; i++) {
      db.push(makeRow(i, i === 4999 ? "needle at the bottom" : `hay ${i}`));
    }
    const { runtime } = makeRuntime(db, { honorTextContains: false });
    const res = await get(
      runtime,
      "/api/memories/browse?type=messages&q=needle&limit=50&offset=0",
    );
    expect(ids(res)).toEqual([makeRow(4999, "").id]);
    expect(res.total).toBe(1);
    expect(res.totalIsExact).toBe(true);
  });

  test("true exhaustion terminates the drain after one call", async () => {
    const db = [makeRow(0, "needle"), makeRow(1, "hay"), makeRow(2, "needle")];
    const { runtime, getMemories } = makeRuntime(db, {
      honorTextContains: false,
    });
    const res = await get(
      runtime,
      "/api/memories/browse?type=messages&q=needle&limit=50&offset=0",
    );
    expect(ids(res)).toHaveLength(2);
    expect(res.total).toBe(2);
    expect(res.totalIsExact).toBe(true);
    expect(getMemories).toHaveBeenCalledTimes(1);
  });

  test("zero-match multi-term q on a bottomless table: bounded work, no prefix re-read, incomplete total", async () => {
    // Every call returns a full page (looks endless) and nothing matches; the
    // adapter cannot apply the any-term predicate so the drain must stop
    // itself and say so. #22066 hit 28 calls / 116,104 rows here.
    const requests: Array<{ offset: number; limit: number }> = [];
    let rows = 0;
    const getMemories = vi.fn(async (params: GetMemoriesParams) => {
      requests.push({ offset: params.offset ?? 0, limit: params.limit });
      rows += params.limit;
      return Array.from({ length: params.limit }, (_, i) => ({
        ...makeRow((params.offset ?? 0) + i, `hay ${i}`),
      }));
    });
    const runtime = {
      agentId: AGENT_ID,
      character: { name: "Eliza" },
      ensureConnection: vi.fn(async () => undefined),
      getMemories,
    } as unknown as AgentRuntime;
    const res = await get(
      runtime,
      "/api/memories/browse?type=messages&q=absent%20needle&limit=50&offset=0",
    );
    expect(ids(res)).toHaveLength(0);
    expect(res.total).toBe(0);
    expect(res.totalIsExact).toBe(false);
    expect(requests.length).toBeLessThanOrEqual(8);
    expect(rows).toBeLessThanOrEqual(50 + 1 + 10_000);
    // Windows are contiguous: each starts where the previous ended.
    let expectedOffset = 0;
    for (const r of requests) {
      expect(r.offset).toBe(expectedOffset);
      expectedOffset += r.limit;
    }
  });
});
