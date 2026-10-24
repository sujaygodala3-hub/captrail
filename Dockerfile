# Research staging frontend. No participant data, preview configuration or secrets copied.
FROM node:22-bookworm-slim AS dependencies
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts && npm cache clean --force
FROM gcr.io/distroless/nodejs22-debian12:nonroot
WORKDIR /app
COPY --from=dependencies /app/node_modules ./node_modules
COPY package.json ./
COPY api ./api
COPY scripts/dev.mjs ./scripts/dev.mjs
COPY research ./research
COPY images ./images
COPY index.html build.html scan.html site.css site.js capture-core.js ./
ENV NODE_ENV=production PORT=3000
EXPOSE 3000
CMD ["scripts/dev.mjs"]
