# ===== Stage 1: Development =====
FROM node:18-slim AS dev
WORKDIR /app
ENV NODE_ENV=dev
# Install dependencies (for dev, includes devDeps)
COPY package.json package-lock.json ./ 
RUN npm install          # Use npm ci if lockfile is present for deterministic install
# Copy all source code
COPY . .
# Expose Vite's dev server port (default 5173)
EXPOSE 5173 
# Start Vite dev server with hot reload (and --host to allow external access)
CMD ["npm", "run", "dev", "--", "--host"]

# ===== Stage 2: Build (Production Build) =====
FROM node:18-slim AS build
WORKDIR /app
ENV NODE_ENV=production
COPY package.json package-lock.json ./
RUN npm ci               # install prod dependencies (ci ensures clean install)
COPY . .
RUN npm run build        # generate optimized production build (outputs to dist/ by default)

# ===== Stage 3: Production (Nginx Serve) =====
FROM nginx:stable-alpine AS production
# Copy build output from previous stage
COPY --from=build /app/dist /usr/share/nginx/html
# Copy custom Nginx config to serve React app and API proxy
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
