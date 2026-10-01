# 前端：Node 构建静态文件，nginx 托管并把 /api 反向代理到后端（SSE 流式输出要关掉缓冲，见 nginx.conf）
FROM node:24-alpine AS build
WORKDIR /app
# 国内网络可在 .env 里设 NPM_REGISTRY（如 https://registry.npmmirror.com）
ARG NPM_REGISTRY=https://registry.npmjs.org
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --registry "$NPM_REGISTRY"
COPY frontend/ ./
RUN npm run typecheck && npm run build

FROM nginx:1.30-alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
