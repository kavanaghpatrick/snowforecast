"""Real local servers that replay recorded provider payloads, or break them on purpose.

Two servers share one rule engine:

- ReplayServer: plain HTTP on 127.0.0.1. Used where the fetcher takes a URL
  (``_fetch`` / ``http_json`` directly, and Open-Meteo through the fetcher's own
  ``OPEN_METEO_URL`` setting).
- InterceptProxy: an HTTPS-intercepting CONNECT proxy on 127.0.0.1, adapted from
  the robustness audit's harness (scratchpad harness/mitm.py). The fetcher runs
  unmodified in a subprocess with the standard ``HTTPS_PROXY`` and
  ``SSL_CERT_FILE`` variables, so its hard-coded provider URLs land here. The
  proxy never connects upstream; an unmatched host gets a 502, and only the
  test CA is trusted, so nothing can reach a real provider.

Every response body is a recorded real payload (tests/fetcher_v2/fixtures/),
or that payload deformed by one of the failure modes below. No mocks, no
patching: these are sockets.

Rule = dict(host=regex, path=regex, mode=..., body=bytes | om=list-of-points, ...)
Modes: replay, status (code), garbage (text), truncate (frac, no_length), slow
(delay_s, before headers), fail_first (n failures with `fail_code`, then replay),
drip (chunk bytes every every_s seconds), redirect (code, location), close (hang up, no answer).
"""

from __future__ import annotations

import datetime as dt
import http.server
import itertools
import json
import os
import re
import socket
import socketserver
import ssl
import threading
import time
import urllib.parse


def om_points_body(points, query):
    """Recorded Open-Meteo points, cycled to the number of requested coordinates.

    Open-Meteo returns one object for one point and a list for several."""
    q = urllib.parse.parse_qs(query)
    n = len(q["latitude"][0].split(","))
    pts = [p for _, p in zip(range(n), itertools.cycle(points))]
    return json.dumps(pts[0] if n == 1 else pts).encode()


class _Engine:
    def __init__(self, rules):
        self.rules = rules
        self.log = []
        self.hits = {}
        self.lock = threading.Lock()

    def match(self, host, path_q):
        for i, r in enumerate(self.rules):
            if re.search(r.get("host", ""), host) and re.search(r.get("path", ""), path_q):
                return i, r
        return None, None


def _make_handler(engine, fixed_host=None):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        host_name = fixed_host

        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urllib.parse.urlsplit(self.path)
            host = self.host_name or (self.headers.get("Host") or "").split(":")[0]
            path_q = u.path + "?" + u.query
            i, rule = engine.match(host, path_q)
            with engine.lock:
                engine.hits[i] = engine.hits.get(i, 0) + 1
                hit = engine.hits[i]
                engine.log.append({"t": time.monotonic(), "host": host, "path": u.path, "query": u.query,
                                   "rule": i, "hit": hit, "ua": self.headers.get("User-Agent"),
                                   "headers": {k.lower(): v for k, v in self.headers.items()}})
            if rule is None:
                return self._send(502, b"replay: no rule", "text/plain")
            mode = rule["mode"]
            ctype = rule.get("ctype", "application/json")
            if mode == "fail_first":
                if hit <= rule["n"]:
                    return self._send(rule.get("fail_code", 500), b'{"error":"replay failure"}',
                                      "application/json", rule.get("headers"))
                mode = "replay"
            if mode == "close":   # accept, read the request, answer nothing, hang up
                self.close_connection = True
                return
            try:
                if mode == "status":
                    return self._send(rule["code"], rule.get("body", b'{"error":"replay status"}'),
                                      ctype, rule.get("headers"))
                if mode == "garbage":
                    return self._send(200, rule["text"].encode(), rule.get("ctype", "text/html"))
                if mode == "redirect":
                    self.send_response(rule.get("code", 302))
                    self.send_header("Location", rule["location"])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = om_points_body(rule["om"], u.query) if "om" in rule else rule["body"]
                if mode == "replay":
                    return self._send(200, body, ctype)
                if mode == "slow":
                    time.sleep(rule["delay_s"])
                    return self._send(200, body, ctype)
                if mode == "drip":
                    # Honest headers, then the body a few bytes at a time: every single
                    # read succeeds well inside a socket timeout.
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.flush()
                    step = rule.get("chunk", 200)
                    for i in range(0, len(body), step):
                        self.wfile.write(body[i:i + step])
                        self.wfile.flush()
                        time.sleep(rule.get("every_s", 0.05))
                    return
                if mode == "truncate":
                    cut = int(len(body) * rule.get("frac", 0.5))
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    if not rule.get("no_length"):
                        self.send_header("Content-Length", str(len(body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(body[:cut])
                    self.wfile.flush()
                    self.close_connection = True
                    return
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError, OSError):
                return
            raise RuntimeError(f"unknown mode {mode}")

        def _send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

    return H


class _Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ReplayServer:
    """Plain HTTP replay server. Use as a context manager; .url is the base URL."""

    def __init__(self, rules):
        self.engine = _Engine(rules)
        self.srv = _Threaded(("127.0.0.1", 0), _make_handler(self.engine))
        self.port = self.srv.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    @property
    def log(self):
        return self.engine.log

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()


# ----------------------------------------------------------------------------- TLS proxy

def _make_ca(cert_dir):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "fetcher_v2 test CA")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=2))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                         content_commitment=False, key_encipherment=False,
                                         data_encipherment=False, key_agreement=False,
                                         encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(key, hashes.SHA256()))
    ca_pem = os.path.join(cert_dir, "ca.pem")
    with open(ca_pem, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return key, cert, ca_pem


def _make_leaf(cert_dir, host, ca_key, ca_cert):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)]))
            .issuer_name(ca_cert.subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=2))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(host)]), critical=False)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
                           critical=False)
            .sign(ca_key, hashes.SHA256()))
    p = os.path.join(cert_dir, re.sub(r"[^A-Za-z0-9.-]", "_", host) + ".pem")
    with open(p, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    return p


class InterceptProxy:
    """HTTPS CONNECT proxy answering every host from the rule list (never upstream)."""

    def __init__(self, rules, cert_dir):
        self.engine = _Engine(rules)
        self.cert_dir = cert_dir
        self.ca_key, self.ca_cert, self.ca_pem = _make_ca(cert_dir)
        self._leaves = {}
        self._leaf_lock = threading.Lock()
        outer = self

        class Proxy(socketserver.BaseRequestHandler):
            def handle(self):
                f = self.request.makefile("rb")
                line = f.readline().decode("latin-1").strip()
                while True:
                    h = f.readline()
                    if h in (b"\r\n", b"\n", b""):
                        break
                parts = line.split()
                if len(parts) < 2 or parts[0] != "CONNECT":
                    outer.engine.log.append({"non_connect": line[:200]})
                    self.request.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
                    return
                host = parts[1].rsplit(":", 1)[0]
                self.request.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(outer._leaf(host))
                try:
                    tls = ctx.wrap_socket(self.request, server_side=True)
                except (ssl.SSLError, OSError):
                    return
                handler = _make_handler(outer.engine, fixed_host=host)
                try:
                    handler(tls, self.client_address, self.server)
                except (ssl.SSLError, OSError):
                    pass
                finally:
                    try:
                        tls.close()
                    except OSError:
                        pass

        class Srv(socketserver.ThreadingMixIn, socketserver.TCPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.srv = Srv(("127.0.0.1", 0), Proxy)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def _leaf(self, host):
        with self._leaf_lock:
            if host not in self._leaves:
                self._leaves[host] = _make_leaf(self.cert_dir, host, self.ca_key, self.ca_cert)
            return self._leaves[host]

    @property
    def log(self):
        return self.engine.log

    def env(self, base_env):
        """Environment for a subprocess that must reach providers only through this proxy."""
        env = {k: v for k, v in base_env.items()
               if k.lower() not in ("http_proxy", "https_proxy", "no_proxy", "all_proxy", "aemet_api_key")}
        env.update({"HTTPS_PROXY": f"http://127.0.0.1:{self.port}",
                    "HTTP_PROXY": f"http://127.0.0.1:{self.port}",
                    "SSL_CERT_FILE": self.ca_pem,
                    # An unresolvable host: even a proxy bypass could never reach real Open-Meteo.
                    "OPEN_METEO_URL": "https://open-meteo.replay.invalid/v1/forecast"})
        return env

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p
