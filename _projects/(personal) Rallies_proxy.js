---
layout: project
date: 2026-07-12
category: personal
---
{% raw %}
// Simple proxy server to bypass CORS for Rallies API
// Requirements: Node.js (https://nodejs.org)
// Run with: node proxy.js
// Then open dashboard.html in your browser

const http = require('http');
const https = require('https');

const PORT = 3131;

const ALLOWED_PATHS = {
  '/arena': 'https://rallies.ai/api/get-arena-data-v3?limit=50',
  '/scans': 'https://rallies.ai/api/get-scans-data-for-mobile',
};

const server = http.createServer((req, res) => {
  // CORS headers — allow any local page to call this proxy
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // Strip query string to match path
  const path = req.url.split('?')[0];
  const targetUrl = ALLOWED_PATHS[path];

  if (!targetUrl) {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Unknown route', available: Object.keys(ALLOWED_PATHS) }));
    return;
  }

  console.log(`[${new Date().toLocaleTimeString()}] Proxying ${path} → ${targetUrl}`);

  https.get(targetUrl, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      'Accept': 'application/json',
    }
  }, (apiRes) => {
    let body = '';
    apiRes.on('data', chunk => body += chunk);
    apiRes.on('end', () => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(body);
    });
  }).on('error', (err) => {
    console.error('Proxy error:', err.message);
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Proxy failed', detail: err.message }));
  });
});

server.listen(PORT, () => {
  console.log(`\n✅ Rallies proxy running at http://localhost:${PORT}`);
  console.log(`   /arena  → Arena data`);
  console.log(`   /scans  → Daily scans`);
  console.log(`\n📂 Now open dashboard.html in your browser`);
  console.log(`   Press Ctrl+C to stop\n`);
});
{% endraw %}