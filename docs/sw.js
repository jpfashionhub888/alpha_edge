// Service Worker for Trading Empire PWA
const CACHE_NAME = 'trading-empire-v1';
const urlsToCache = [
    '/alpha_edge/master.html',
    '/alpha_edge/manifest.json',
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => response || fetch(event.request))
    );
});