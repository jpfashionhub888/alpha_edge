// AlphaEdge Terminal - Service Worker v5 -- KILL SWITCH
//
// docs/app.html is now just a redirect stub to the live dashboard
// (alphaedgetrading.duckdns.org). The old v4 worker above cached the
// app shell cache-first, which meant browsers that had already
// installed it kept serving the OLD app.html straight from cache and
// never saw the new redirect stub at all -- the exact "same frozen
// page forever" failure mode from earlier tonight, just relocated to
// a service worker instead of a data file. This version deletes every
// cache, unregisters itself, and gets out of the way on the very next
// load, so every future request goes straight to the network.
self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.map(k => caches.delete(k))))
      .then(() => self.registration.unregister())
      .then(() => self.clients.matchAll())
      .then(clients => clients.forEach(client => client.navigate(client.url)))
  );
});