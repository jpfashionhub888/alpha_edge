// AlphaEdge Terminal - Service Worker v4
const CACHE_NAME = 'alphaedge-v4';
const PRECACHE = [
  '/alpha_edge/app.html',
  '/alpha_edge/manifest.json',
  '/alpha_edge/icons/alpha_edge_logo.png',
  '/alpha_edge/icon-192.png',
  '/alpha_edge/icon-512.png'
];

// Install: pre-cache the app shell only
self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(PRECACHE))
  );
});

// Activate: remove ALL old caches immediately
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => {
        console.log('[SW] Deleting old cache:', k);
        return caches.delete(k);
      }))
    ).then(() => self.clients.claim())
  );
});

// Fetch strategy:
// - JSON data files → ALWAYS network (never cached, always fresh)
// - App shell       → cache-first with network fallback
self.addEventListener('fetch', event => {
  const url = event.request.url;

  // NEVER cache data files - always fetch fresh from network
  if (url.includes('/data/') || url.includes('.json')) {
    event.respondWith(
      fetch(event.request, { cache: 'no-store' })
        .catch(() => new Response('{"error":"offline"}', {
          headers: { 'Content-Type': 'application/json' }
        }))
    );
    return;
  }

  // App shell: cache-first
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(response => {
        if (response && response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(c => c.put(event.request, clone));
        }
        return response;
      });
    })
  );
});