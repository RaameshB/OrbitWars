const CORS = {
  'Access-Control-Allow-Origin':  '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age':       '86400',
};

const GITHUB_OWNER = 'RaameshB';
const GITHUB_REPO  = 'OrbitWars';
const R2_KEY       = 'rl_v25/hof_index.json';

// UUID-ish safety check — only allow alphanumeric + hyphens, 8-64 chars
function safeReqId(s) {
  return typeof s === 'string' && /^[a-zA-Z0-9_-]{8,64}$/.test(s) ? s : null;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS });
    }

    // GET /hof-index — proxy hof_index.json from R2
    if (request.method === 'GET' && url.pathname === '/hof-index') {
      const obj = await env.R2_BUCKET.get(R2_KEY);
      if (!obj) {
        return new Response(JSON.stringify({ error: 'Index not found' }), {
          status: 404,
          headers: { ...CORS, 'Content-Type': 'application/json' },
        });
      }
      const body = await obj.text();
      return new Response(body, {
        headers: {
          ...CORS,
          'Content-Type': 'application/json',
          'Cache-Control': 'no-store',
        },
      });
    }

    // GET /replay-status/:reqId — check if replay is ready (meta.json exists in R2)
    const statusMatch = url.pathname.match(/^\/replay-status\/(.+)$/);
    if (request.method === 'GET' && statusMatch) {
      const reqId = safeReqId(statusMatch[1]);
      if (!reqId) {
        return new Response(JSON.stringify({ error: 'Invalid reqId' }), {
          status: 400, headers: { ...CORS, 'Content-Type': 'application/json' },
        });
      }
      const obj = await env.R2_BUCKET.get(`replays/${reqId}/meta.json`);
      if (!obj) {
        return new Response(JSON.stringify({ ready: false }), {
          status: 404, headers: { ...CORS, 'Content-Type': 'application/json' },
        });
      }
      const meta = await obj.text();
      return new Response(JSON.stringify({ ready: true, meta: JSON.parse(meta) }), {
        headers: { ...CORS, 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
      });
    }

    // GET /replay/:reqId — serve replay HTML from R2
    const replayMatch = url.pathname.match(/^\/replay\/(.+)$/);
    if (request.method === 'GET' && replayMatch) {
      const reqId = safeReqId(replayMatch[1]);
      if (!reqId) {
        return new Response('Invalid reqId', { status: 400, headers: CORS });
      }
      const obj = await env.R2_BUCKET.get(`replays/${reqId}/replay.html`);
      if (!obj) {
        return new Response('Replay not found', { status: 404, headers: CORS });
      }
      // Stream R2 body directly — avoids buffering large HTML into a string,
      // which can exceed the Workers free-tier CPU time limit.
      return new Response(obj.body, {
        headers: {
          ...CORS,
          'Content-Type': 'text/html; charset=utf-8',
          'Cache-Control': 'no-store',
        },
      });
    }

    // POST / — trigger GitHub Action with {players, agents, req_id}
    if (request.method === 'POST' && url.pathname === '/') {
      let payload;
      try {
        payload = await request.json();
      } catch {
        return new Response(JSON.stringify({ error: 'Invalid JSON' }), {
          status: 400,
          headers: { ...CORS, 'Content-Type': 'application/json' },
        });
      }

      const players = payload.players ?? 4;
      const agents  = Array.isArray(payload.agents) ? payload.agents : [];
      const req_id  = safeReqId(payload.req_id) ?? '';

      const ghRes = await fetch(
        `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/dispatches`,
        {
          method: 'POST',
          headers: {
            'Accept':        'application/vnd.github.v3+json',
            'Authorization': `token ${env.GITHUB_PAT}`,
            'User-Agent':    'Cloudflare-Worker/OrbitWars',
            'Content-Type':  'application/json',
          },
          body: JSON.stringify({
            event_type:     'trigger-simulation',
            client_payload: { players, agents, req_id },
          }),
        }
      );

      if (!ghRes.ok) {
        const detail = await ghRes.text();
        return new Response(JSON.stringify({ error: 'GitHub dispatch failed', detail }), {
          status: 502,
          headers: { ...CORS, 'Content-Type': 'application/json' },
        });
      }

      return new Response(
        JSON.stringify({ success: true, players, agents, req_id }),
        { headers: { ...CORS, 'Content-Type': 'application/json' } }
      );
    }

    return new Response('Not found', { status: 404, headers: CORS });
  },
};
