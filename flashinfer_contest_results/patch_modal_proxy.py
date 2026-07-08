# 功能：为 Modal 1.4.3 的 grpclib channel 增加 HTTPS proxy CONNECT 支持。
# 参数：
#   无。脚本会自动定位当前 Python 环境中的 modal/_utils/grpc_utils.py。
# 示例：
#   cd flashinfer_contest_results
#   .venv/bin/python patch_modal_proxy.py

from __future__ import annotations

from pathlib import Path

import modal._utils.grpc_utils as grpc_utils


PROXY_BLOCK = '''
def _https_proxy_url() -> str | None:
    return os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")


def _connect_via_http_proxy(proxy_url: str, host: str, port: int) -> socket.socket:
    parsed = urllib.parse.urlparse(proxy_url)
    if parsed.scheme not in ("http", ""):
        raise RuntimeError(f"Unsupported HTTPS proxy scheme for Modal gRPC: {parsed.scheme!r}")
    proxy_host = parsed.hostname
    proxy_port = parsed.port or 8080
    if proxy_host is None:
        raise RuntimeError(f"Invalid HTTPS proxy URL for Modal gRPC: {proxy_url!r}")

    sock = socket.create_connection((proxy_host, proxy_port), timeout=30)
    target = f"{host}:{port}"
    request = (
        f"CONNECT {target} HTTP/1.1\\r\\n"
        f"Host: {target}\\r\\n"
        "Proxy-Connection: Keep-Alive\\r\\n"
        "\\r\\n"
    ).encode("ascii")
    sock.sendall(request)

    response = b""
    while b"\\r\\n\\r\\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            sock.close()
            raise RuntimeError("HTTPS proxy closed while opening Modal gRPC tunnel")
        response += chunk
        if len(response) > 65536:
            sock.close()
            raise RuntimeError("HTTPS proxy response too large while opening Modal gRPC tunnel")
    status_line = response.split(b"\\r\\n", 1)[0]
    if b" 200 " not in status_line:
        sock.close()
        raise RuntimeError(
            "HTTPS proxy refused Modal gRPC tunnel: "
            + status_line.decode("latin1", errors="replace")
        )
    sock.setblocking(False)
    return sock


class _HttpProxyChannel(grpclib.client.Channel):
    async def _create_connection(self) -> H2Protocol:
        proxy_url = _https_proxy_url()
        if self._path is None and self._ssl is not None and proxy_url:
            sock = await self._loop.run_in_executor(
                None,
                _connect_via_http_proxy,
                proxy_url,
                self._host,
                self._port,
            )
            _, protocol = await self._loop.create_connection(
                self._protocol_factory,
                sock=sock,
                ssl=self._ssl,
                server_hostname=(
                    self._config.ssl_target_name_override or self._host
                ),
            )
            return protocol
        return await super()._create_connection()


'''.lstrip()


def patch_modal_grpc_utils() -> None:
    """给当前环境的 Modal grpc_utils.py 打补丁；无输入，直接修改 site-packages 文件。"""
    path = Path(grpc_utils.__file__).resolve()
    text = path.read_text()
    if "_HttpProxyChannel" in text:
        print(f"[setup] Modal proxy patch already present: {path}")
        return

    insert_anchor = "custom_detail_codec = CustomProtoStatusDetailsCodec()\n\n\n"
    replace_from = (
        "        channel = grpclib.client.Channel("
        "host, port, ssl=ssl, config=config, status_details_codec=custom_detail_codec)"
    )
    replace_to = (
        "        channel = _HttpProxyChannel("
        "host, port, ssl=ssl, config=config, status_details_codec=custom_detail_codec)"
    )
    if insert_anchor not in text:
        raise RuntimeError(f"Modal proxy patch anchor not found in {path}: custom_detail_codec")
    if replace_from not in text:
        raise RuntimeError(f"Modal proxy patch anchor not found in {path}: channel constructor")

    text = text.replace(insert_anchor, insert_anchor + PROXY_BLOCK, 1)
    text = text.replace(replace_from, replace_to, 1)
    path.write_text(text)
    print(f"[setup] applied Modal proxy patch: {path}")


if __name__ == "__main__":
    patch_modal_grpc_utils()
