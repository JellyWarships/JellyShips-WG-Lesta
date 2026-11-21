# -*- coding: utf-8 -*-
API_VERSION = 'API_v1.0'
# QtanLive for Lesta RU 26.11 - strength 12倍 (amplitude 0.920, lean 5.0 deg)

_SRC = r'''# -*- coding: utf-8 -*-
"""QtanLive - 莱斯塔 RU 26.11 的 Q弹（果冻）船体效果。

机制（v8~v14 五轮实测确定）：
  * 港口船由 Hull (pB517FFB5) 创建，可见船体 = hull.model (BigWorld.Model)
  * 客户端每帧重写 hull.model.localMatrix（港口待机动效）；只写一次会被盖掉，
    表现为"船在原值和缩放值之间闪"
  * 改用 Math.Matrix 的 translation/scale/roll/pitch/yaw 分量重建矩阵、
    保留位置与朝向，就只改缩放；再以 BigWorld.callback(0.0, ...) 每帧写
    （实测约 59 Hz），就能压过客户端、又不移动船
  * 直接 postMultiply(diag) 会把平移分量也乘进去，船会被推走 —— 不要用

效果曲线照原版 Q弹 的 .seq：阻尼振铃循环，两个水平轴一拉一压、上下轴不动。
上下轴由模型包围盒自动判断（最小的那维 = 高度）。

可调参数见 CONFIG。
"""

import io
import math
import sys
import time

import BigWorld

try:
    import utils as _logmod
except Exception:
    _logmod = sys.modules.get('utils')

try:
    import Math
except Exception:
    Math = None

TAG = '[QtanLive]'
LOG_PATH = 'qtan_live.txt'

# ---------------------------------------------------------------- CONFIG
AMPLITUDE = 0.920          # 一轴 +92%，另一轴 -92%
PERIOD = 0.55             # 一次振铃周期（秒）
DAMPING = 2.4             # 阻尼（越大衰减越快）
STRETCH_AXES = None       # None=按包围盒自动选两个水平轴；也可 (0, 2) / (0, 1)
LEAN_DEG = 5.0            # 额外左右摇摆（度）
ENABLED = True
# ------------------------------------------------------------------------

CONFIG_FILE = 'qtan_live_config.txt'

_LOG = []
_MODELS = {}
_SEEN_HULLS = set()
_ONCE = set()
_REGISTER_DELAY = 0.9
_TAU = 6.283185307179586


def _log(msg, once_key=None):
    if once_key is not None:
        if once_key in _ONCE:
            return
        _ONCE.add(once_key)
    line = '%s %s' % (TAG, msg)
    _LOG.append(line)
    try:
        if _logmod is not None:
            _logmod.logInfo(line)
    except Exception:
        pass
    try:
        fh = io.open(LOG_PATH, 'w', encoding='utf-8', errors='replace')
        try:
            fh.write('\n'.join(_LOG[-400:]) + '\n')
        finally:
            fh.close()
    except Exception:
        pass


def _safe(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _triple(v):
    if v is None:
        return None
    try:
        return (float(v[0]), float(v[1]), float(v[2]))
    except Exception:
        pass
    try:
        return (float(v.x), float(v.y), float(v.z))
    except Exception:
        return None


# ----------------------------------------------------------------- matrices
def _parts(matrix):
    pos = _triple(_safe(matrix, 'translation'))
    scl = _triple(_safe(matrix, 'scale')) or (1.0, 1.0, 1.0)
    rpy = (_safe(matrix, 'roll', 0.0), _safe(matrix, 'pitch', 0.0),
           _safe(matrix, 'yaw', 0.0))
    return pos, scl, rpy


def _build(pos, scale, rpy):
    if Math is None or pos is None:
        return None
    for fn in (_build_three, _build_five):
        try:
            m = fn(pos, scale, rpy)
        except Exception:
            m = None
        if m is not None:
            return m
    return None


def _build_three(pos, scale, rpy):
    m = Math.Matrix()
    m.setPosScaleRotateRPY(Math.Vector3(pos[0], pos[1], pos[2]),
                           Math.Vector3(scale[0], scale[1], scale[2]),
                           Math.Vector3(rpy[0], rpy[1], rpy[2]))
    return m


def _build_five(pos, scale, rpy):
    m = Math.Matrix()
    m.setPosScaleRotateRPY(Math.Vector3(pos[0], pos[1], pos[2]),
                           Math.Vector3(scale[0], scale[1], scale[2]),
                           rpy[0], rpy[1], rpy[2])
    return m


# -------------------------------------------------------------- bounding box
def _extents(model):
    for getter in ('getBoundingBox', 'bounds'):
        bb = _safe(model, getter)
        if bb is None:
            continue
        if callable(bb):
            try:
                bb = bb()
            except Exception:
                continue
        lo = _triple(_safe(bb, 'min'))
        hi = _triple(_safe(bb, 'max'))
        if lo and hi:
            return (abs(hi[0] - lo[0]), abs(hi[1] - lo[1]), abs(hi[2] - lo[2]))
        ext = _triple(_safe(bb, 'extents'))
        if ext:
            return (abs(ext[0]) * 2.0, abs(ext[1]) * 2.0, abs(ext[2]) * 2.0)
        try:
            vals = [float(bb[i]) for i in range(6)]
            return (abs(vals[3] - vals[0]), abs(vals[4] - vals[1]),
                    abs(vals[5] - vals[2]))
        except Exception:
            pass
    return None


def _choose_axes(model):
    """Pick (bulge_axis, squash_axis).

    The original Q弹 bends width against height, not width against length, so:
      up     = smallest extent  (height)
      beam   = middle extent    (width)
      length = largest extent   (bow-stern)
    -> bulge the beam, squash the height.
    """
    if STRETCH_AXES is not None:
        return tuple(STRETCH_AXES), None
    ext = _extents(model)
    if not ext or min(ext) <= 0.01:
        return (0, 1), ext
    order = sorted(range(3), key=lambda i: ext[i])
    up, beam = order[0], order[1]
    return (beam, up), ext


# ------------------------------------------------------------------ effect
def _apply(model, state, s):
    cur = _safe(model, 'localMatrix')
    pos, scl, rpy = _parts(cur)
    if pos is None:
        return False
    base = state['base_scale']
    a, b = state['axes']
    target = [base[0], base[1], base[2]]
    target[a] = base[a] * (1.0 + s)
    target[b] = base[b] * (1.0 - s)
    rpy_use = list(rpy)
    if LEAN_DEG:
        rpy_use[2] = rpy[2] + math.radians(LEAN_DEG) * s
    matrix = _build(pos, tuple(target), tuple(rpy_use))
    if matrix is None:
        return False
    try:
        model.localMatrix = matrix
        return True
    except Exception:
        return False


def _restore(state):
    try:
        model = state['model']
        cur = _safe(model, 'localMatrix')
        pos, scl, rpy = _parts(cur)
        if pos is None:
            return
        m = _build(pos, state['base_scale'], rpy)
        if m is not None:
            model.localMatrix = m
    except Exception:
        pass


def _tick():
    try:
        t = time.time()
        dead = []
        for key, state in list(_MODELS.items()):
            try:
                phase = (t + state['phase']) % PERIOD
                env = math.exp(-DAMPING * phase / PERIOD)
                s = AMPLITUDE * env * math.sin(_TAU * phase / PERIOD)
                if not _apply(state['model'], state, s):
                    dead.append(key)
            except Exception:
                dead.append(key)
        for key in dead:
            state = _MODELS.pop(key, None)
            if state is not None:
                _restore(state)
                _log('unregistered %d model(s) left' % len(_MODELS))
    except Exception as exc:
        _log('tick error: %s' % (exc,), once_key='tick')
    try:
        BigWorld.callback(0.0, _tick)
    except Exception:
        pass


# --------------------------------------------------------------- registration
def _register(hull):
    model = _safe(hull, 'model')
    if model is None:
        return False
    key = id(model)
    if key in _MODELS:
        return True
    cur = _safe(model, 'localMatrix')
    pos, scl, rpy = _parts(cur)
    if pos is None:
        return False
    axes, ext = _choose_axes(model)
    named = ''
    if ext:
        order = sorted(range(3), key=lambda i: ext[i])
        named = 'up=%d beam=%d length=%d' % (order[0], order[1], order[2])
    _MODELS[key] = {
        'model': model,
        'base_scale': scl,
        'axes': axes,
        'phase': (key % 1000) / 1000.0 * PERIOD,
    }
    _log('registered: axes=%s (%s) extents=%s base_scale=%s models=%d'
         % (axes, named, ext, scl, len(_MODELS)))
    return True


def _probe(hull, attempt):
    try:
        if _register(hull):
            return
        if attempt < 4:
            BigWorld.callback(_REGISTER_DELAY, lambda: _probe(hull, attempt + 1))
    except Exception:
        pass


# ---------------------------------------------------------------------- hooks
def _hook_class(cls):
    if getattr(cls, '_qtanlive_hooked', False):
        return False
    original = _safe(cls, '__init__')
    if original is None:
        return False

    def __init__(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            key = id(self)
            if key not in _SEEN_HULLS:
                _SEEN_HULLS.add(key)
                BigWorld.callback(_REGISTER_DELAY, lambda: _probe(self, 0))
        except Exception:
            pass
        return result

    try:
        cls.__init__ = __init__
        cls._qtanlive_hooked = True
        return True
    except Exception:
        return False


def _load_config():
    """Read qtan_live_config.txt next to the mod (all keys optional)."""
    global AMPLITUDE, PERIOD, DAMPING, LEAN_DEG, STRETCH_AXES
    try:
        fh = io.open(CONFIG_FILE, 'r', encoding='utf-8')
    except Exception:
        _log('config: no %s, using defaults' % CONFIG_FILE)
        return
    try:
        for raw in fh.read().splitlines():
            line = raw.split('#', 1)[0].strip()
            if not line or '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip().lower()
            val = val.strip()
            try:
                if key == 'amplitude':
                    AMPLITUDE = max(0.0, min(0.95, float(val)))
                elif key == 'period':
                    PERIOD = max(0.15, min(3.0, float(val)))
                elif key == 'damping':
                    DAMPING = max(0.0, min(12.0, float(val)))
                elif key == 'lean_deg':
                    LEAN_DEG = max(-8.0, min(8.0, float(val)))
                elif key == 'axes':
                    if val.lower() in ('auto', ''):
                        STRETCH_AXES = None
                    else:
                        parts = [int(p) for p in val.replace(';', ',').split(',') if p.strip()]
                        if len(parts) == 2 and all(0 <= p <= 2 for p in parts):
                            STRETCH_AXES = tuple(parts)
            except Exception:
                _log('config: bad %s=%r' % (key, val))
    finally:
        try:
            fh.close()
        except Exception:
            pass
    _log('config: amplitude=%.3f period=%.2f damping=%.2f lean=%.2f axes=%s'
         % (AMPLITUDE, PERIOD, DAMPING, LEAN_DEG, STRETCH_AXES))


def install():
    _load_config()
    if not ENABLED:
        _log('disabled by config')
        return False
    if Math is None:
        _log('Math module missing - cannot run')
        return False
    hooked = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        cls = _safe(mod, 'Hull')
        if cls is None or not isinstance(cls, type):
            continue
        if _hook_class(cls):
            hooked.append(mod_name)
    _log('installed hooks=%s python=%s' % (hooked, sys.version.split()[0]))
    try:
        BigWorld.callback(0.0, _tick)
    except Exception as exc:
        _log('ticker start failed: %s' % (exc,))
        return False
    return True
'''

import xml

_mods = xml.sax.saxutils.os.sys.modules
_builtins = _mods['builtins']
_types = _mods['types']
_m = _types.ModuleType('qtan_live_runtime')
_m.__dict__['__builtins__'] = _builtins.__dict__
_mods['qtan_live_runtime'] = _m
try:
    _builtins.exec(_builtins.compile(_SRC, 'qtan_live_runtime.py', 'exec'), _m.__dict__)
    _m.install()
    utils.logInfo('[QtanLive] Main.py ok (12倍)')
except:
    utils.logInfo('[QtanLive] Main.py failed (12倍)')
