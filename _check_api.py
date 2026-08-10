"""Quick API introspection for WebSocket/Frame events."""
import inspect, re
from playwright.async_api import Page, WebSocket, Frame, BrowserContext, ElementHandle

# WebSocket events
ws_src = inspect.getsource(WebSocket)
ws_events = re.findall(r'"([a-z]+)"', ws_src)
ws_events = [e for e in ws_events if e in ("framesent", "framereceived", "close")]
print("WebSocket events:", sorted(set(ws_events)))

# Page events
pg_src = inspect.getsource(Page)
pg_events = re.findall(r'"([a-z_]+)"', pg_src)
pg_events = [e for e in pg_events if e in (
    "framesent", "framereceived", "socketopen",
    "frameattached", "framenavigated", "response", "requestfailed",
    "websocket", "frame"
)]
print("Page events:", sorted(set(pg_events)))
print("expect_websocket:", hasattr(Page, "expect_websocket"))
print("route_web_socket:", hasattr(Page, "route_web_socket"))

# Frame methods
fmethods = [m for m in dir(Frame) if 'eval' in m or 'selector' in m or 'wait' in m]
print("Frame eval/selector/wait methods:", sorted(fmethods))

# ElementHandle eval
emethods = [m for m in dir(ElementHandle) if 'eval' in m]
print("ElementHandle eval methods:", sorted(emethods))

# Context signatures
print("\nstorage_state:", inspect.signature(BrowserContext.storage_state))
print("add_cookies:", inspect.signature(BrowserContext.add_cookies))
