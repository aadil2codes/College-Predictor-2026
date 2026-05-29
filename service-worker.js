const CACHE_NAME = 'campus-cipher-pwa-v2';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/favicon.png',
  '/icon-192.png',
  '/icon-512.png',
  'https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap',
  'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/jspdf-autotable/3.5.31/jspdf.plugin.autotable.min.js'
];

// Install event - Pre-cache static UI assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[Service Worker] Pre-caching static assets');
        // Cache assets, allowing individual failures if external CDNs are blocked or slow
        return Promise.allSettled(
          STATIC_ASSETS.map(asset => {
            return cache.add(asset).catch(err => {
              console.warn(`[Service Worker] Failed to pre-cache asset: ${asset}`, err);
            });
          })
        );
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - Clean up legacy caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cache => {
          if (cache !== CACHE_NAME) {
            console.log('[Service Worker] Clearing old cache:', cache);
            return caches.delete(cache);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch event - Cache-first with stale-while-revalidate for static, network-only for APIs
self.addEventListener('fetch', event => {
  const requestUrl = new URL(event.request.url);

  // Network-Only strategy for dynamic prediction and AI chatbot APIs (Do NOT cache dynamic results)
  if (
    requestUrl.pathname === '/predict' ||
    requestUrl.pathname === '/search' ||
    requestUrl.pathname === '/lowest_cutoff' ||
    requestUrl.pathname === '/last_cutoffs' ||
    requestUrl.pathname === '/api/colleges' ||
    requestUrl.pathname === '/api/branches' ||
    requestUrl.pathname === '/api/chat' ||
    event.request.method !== 'GET'
  ) {
    event.respondWith(
      fetch(event.request).catch(err => {
        console.log('[Service Worker] API Fetch failed (offline):', err);
        
        // Return a structured offline JSON fallback if it's an API request
        const acceptHeader = event.request.headers.get('accept') || '';
        if (acceptHeader.includes('application/json') || event.request.url.includes('/api/')) {
          return new Response(JSON.stringify({
            error: "offline",
            response: "You are currently offline. Please check your internet connection to run college predictions or speak with CampusCipher AI!"
          }), {
            headers: { 'Content-Type': 'application/json' }
          });
        }
        
        // Throw normal error if not JSON API
        throw err;
      })
    );
    return;
  }

  // Stale-While-Revalidate caching strategy for static pages, fonts, CSS, and scripts
  event.respondWith(
    caches.match(event.request)
      .then(cachedResponse => {
        if (cachedResponse) {
          // Serve from cache immediately, then fetch update in background
          fetch(event.request).then(networkResponse => {
            if (networkResponse.status === 200) {
              caches.open(CACHE_NAME).then(cache => cache.put(event.request, networkResponse));
            }
          }).catch(err => {
            console.log('[Service Worker] Stale update failed (offline mode)', err);
          });

          return cachedResponse;
        }

        // Fetch from network if not cached
        return fetch(event.request).then(response => {
          // Dynamically cache newly accessed static GET assets (not JSON APIs)
          if (
            response.status === 200 && 
            event.request.method === 'GET' && 
            !response.headers.get('content-type')?.includes('application/json')
          ) {
            const responseClone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(event.request, responseClone));
          }
          return response;
        });
      })
  );
});
