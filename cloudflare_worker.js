export default {
  async fetch(request, env, ctx) {
    // 1. Handle CORS Preflight (OPTIONS request)
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*", // Or restrict to your github pages URL
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
          "Access-Control-Max-Age": "86400",
        },
      });
    }

    // 2. Only allow POST requests
    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    // 3. Parse JSON payload from Frontend
    let payload;
    try {
      payload = await request.json();
    } catch (err) {
      return new Response("Invalid JSON", { status: 400 });
    }
    
    // Default to 4 players if not specified
    const players = payload.players || 4;

    // 4. Securely Trigger GitHub Actions via API
    // Ensure you have set the GITHUB_PAT secret in your Cloudflare dashboard!
    const GITHUB_PAT = env.GITHUB_PAT; 
    
    // REPLACE THESE WITH YOUR REPO DETAILS
    const GITHUB_OWNER = "RaameshB";
    const GITHUB_REPO = "OrbitWars";

    const githubUrl = `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/dispatches`;

    const githubResponse = await fetch(githubUrl, {
      method: 'POST',
      headers: {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': `token ${GITHUB_PAT}`,
        'User-Agent': 'Cloudflare-Worker'
      },
      body: JSON.stringify({
        event_type: "trigger-simulation",
        client_payload: {
          players: players
        }
      })
    });

    if (!githubResponse.ok) {
      const errorText = await githubResponse.text();
      console.error("GitHub API Error:", errorText);
      return new Response(`Failed to trigger GitHub Action: ${githubResponse.status}`, { 
        status: 500,
        headers: { "Access-Control-Allow-Origin": "*" }
      });
    }

    // 5. Return success to frontend
    return new Response(JSON.stringify({ success: true, message: `Triggered ${players}-way simulation` }), {
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*"
      }
    });
  },
};
