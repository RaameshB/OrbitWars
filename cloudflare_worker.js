const CORS = {
  'Access-Control-Allow-Origin':  '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age':       '86400',
};

const GITHUB_OWNER = 'RaameshB';
const GITHUB_REPO  = 'OrbitWars';
const R2_KEY       = 'rl_v20/hof_index.json';

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

    // POST / — trigger GitHub Action with {players, agents}
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
            client_payload: { players, agents },
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
        JSON.stringify({ success: true, players, agents }),
        { headers: { ...CORS, 'Content-Type': 'application/json' } }
      );
    }

    return new Response('Not found', { status: 404, headers: CORS });
  },
};
