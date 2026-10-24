# Official Node security release. Serving image contains only its ELF closure and app.
FROM node:24.22.1-bookworm-slim AS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts && npm cache clean --force
COPY scripts/build-node-runtime.mjs ./scripts/build-node-runtime.mjs
RUN node scripts/build-node-runtime.mjs
FROM scratch
COPY --from=dependencies /runtime-root/ /
WORKDIR /app
COPY --from=dependencies /app/node_modules ./node_modules
COPY package.json ./
COPY api ./api
COPY scripts/dev.mjs ./scripts/dev.mjs
COPY research ./research
COPY images ./images
COPY index.html build.html scan.html site.css site.js capture-core.js ./
USER 10001:10001
ENV NODE_ENV=production PORT=3000
EXPOSE 3000
ENTRYPOINT ["/usr/local/bin/node"]
CMD ["scripts/dev.mjs"]
