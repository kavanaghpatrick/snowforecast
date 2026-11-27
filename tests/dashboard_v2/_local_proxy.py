"""A real local HTTP CONNECT proxy for the GitHub read, on 127.0.0.1.

Modes:
  refuse - answers every CONNECT with "502 Bad Gateway" and closes (GitHub down).
  stall  - accepts the tunnel, sends "200 Connection established", then never
           sends a byte (GitHub accepts the connection and hangs).

The dashboard's urllib honours HTTPS_PROXY, so pointing a subprocess at this
server sends data._read_github() here instead of to raw.githubusercontent.com.
Nothing is patched; the real code meets a real misbehaving peer.
"""

from __future__ import annotations

import socket
import threading


class LocalProxy:
    def __init__(self, mode, hold_s=60):
        self.mode, self.hold_s = mode, hold_s
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.connects = 0
        self._stop = threading.Event()
        self._conns = []
        threading.Thread(target=self._serve, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def _serve(self):
        while not self._stop.is_set():
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            self._conns.append(c)
            threading.Thread(target=self._handle, args=(c,), daemon=True).start()

    def _handle(self, c):
        try:
            req = b""
            while b"\r\n\r\n" not in req:
                chunk = c.recv(4096)
                if not chunk:
                    return
                req += chunk
            if req.startswith(b"CONNECT"):
                self.connects += 1
            if self.mode == "refuse":
                c.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
                c.close()
                return
            c.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self._stop.wait(self.hold_s)
        except OSError:
            pass
        finally:
            try:
                c.close()
            except OSError:
                pass

    def close(self):
        self._stop.set()
        for c in self._conns:
            try:
                c.close()
            except OSError:
                pass
        self.sock.close()
