// AlphaEdge Terminal - Service Worker v2
const CACHE_NAME = 'alphaedge-v2';
const PRECACHE = [
  '/alpha_edge/app.html',
  '/alpha_edge/manifest.json',
  '/alpha_edge/icons/alpha_edge_logo.png',
  '/alpha_edge/icon-192.png',
  '/alpha_edge/icon-512.png'
];

// Install: pre-cache the app shell
self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(PRECACHE))
  );
});

// Activate: remove old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Fetch: network-first for JSON data, cache-first for app shell
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Always fetch live data from network
  if (url.pathname.includes('/data/') || url.pathname.includes('.json')) {
    event.respondWith(
      fetch(event.request).catch(() => caches.match(event.request))
    );
    return;
  }

  // App shell: cache-first with network fallback
  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(response => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return response;
      });
    })
  );
});