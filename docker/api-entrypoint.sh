#!/bin/sh
set -e
# Chroma 打开库时要写 SQLite，而宿主机的 chroma_db 是只读挂进来的（/kb/chroma_db）：
# 每次启动复制一份到容器里用（约 7 MB），容器怎么写都碰不到宿主机上的库。
if [ -d /kb/chroma_db ]; then
  rm -rf /app/chroma_db
  cp -r /kb/chroma_db /app/chroma_db
fi
exec "$@"
