# model.py
import cv2
import math
from typing import Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import MobileNet_V3_Small_Weights

# -----------------------------
# Bound of campus
# -----------------------------
LAT_MIN = 39.95009994506836
LAT_MAX = 39.9530029296875
LON_MIN = -75.1928939819336
LON_MAX = -75.18990325927734

# Grid separating campus region
GRID_H = 24
GRID_W = 20

SOFTMAX_TEMP = 0.85

INPUT_SIZE = 224
RESIZE_SHORT = 256

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_MEAN_T = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
_STD_T = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.tensor([p], dtype=torch.float32))
        else:
            self.register_buffer("p", torch.tensor([p], dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0, max=6.0)
        x = F.relu(x) + self.eps
        x = x.pow(p)  # Apply generalized mean
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.pow(1.0 / p)
        return x


def _build_mobilenet_v3_small():
    return torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)


def _deg2rad(x: torch.Tensor) -> torch.Tensor:
    return x * (math.pi / 180.0)


def haversine_m(pred_deg: torch.Tensor, tgt_deg: torch.Tensor) -> torch.Tensor:
    R = 6_371_000.0

    lat1 = _deg2rad(pred_deg[:, 0])
    lon1 = _deg2rad(pred_deg[:, 1])
    lat2 = _deg2rad(tgt_deg[:, 0])
    lon2 = _deg2rad(tgt_deg[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat * 0.5) ** 2 + torch.cos(lat1) * torch.cos(lat2) * (torch.sin(dlon * 0.5) ** 2)
    a = a.clamp(0.0, 1.0)
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt((1.0 - a).clamp(0.0, 1.0)))
    return R * c


class Model(nn.Module):
    """
    Submission entry-point requirements:
      - instantiable without args
      - backend calls predict(batch) if available else forward(batch)
      - outputs [lat, lon] in raw degrees (not normalized)
    """

    def __init__(self):
        super().__init__()
        self.lat_min = float(LAT_MIN)
        self.lat_max = float(LAT_MAX)
        self.lon_min = float(LON_MIN)
        self.lon_max = float(LON_MAX)

        self.grid_h = int(GRID_H)
        self.grid_w = int(GRID_W)
        self.num_cells = self.grid_h * self.grid_w

        dlat = (self.lat_max - self.lat_min) / float(self.grid_h)
        dlon = (self.lon_max - self.lon_min) / float(self.grid_w)

        lat_edges = torch.linspace(self.lat_min, self.lat_max, self.grid_h + 1, dtype=torch.float32)
        lon_edges = torch.linspace(self.lon_min, self.lon_max, self.grid_w + 1, dtype=torch.float32)
        lat_centers = (lat_edges[:-1] + lat_edges[1:]) * 0.5
        lon_centers = (lon_edges[:-1] + lon_edges[1:]) * 0.5

        yy, xx = torch.meshgrid(lat_centers, lon_centers, indexing="ij")
        centers = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)  # [K,2]

        self.register_buffer("cell_centers_deg", centers)
        self.register_buffer("cell_size_deg", torch.tensor([dlat, dlon], dtype=torch.float32))

        base = _build_mobilenet_v3_small()
        self.backbone_features = base.features
        self.gem = GeM(p=3.0, learn_p=True)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
            feat = self.backbone_features(dummy)
            pooled = self.gem(feat).flatten(1)
            feat_dim = pooled.shape[1]

        self.dropout = nn.Dropout(p=0.15)
        self.cell_head = nn.Linear(feat_dim, self.num_cells)
        self.offset_head = nn.Linear(feat_dim, self.num_cells * 2)
        self.softmax_temp = float(SOFTMAX_TEMP)

    # -------- robust image decoding (paths, PIL, HF dicts with bytes/path) --------
    def _normalize_np_image(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.ndim != 3 or arr.shape[2] != 3:
            return np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)

        if arr.dtype != np.uint8:
            amax = float(np.max(arr)) if arr.size else 1.0
            if amax <= 1.5:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _to_rgb_uint8(self, item: Any) -> np.ndarray:
        if isinstance(item, str):
            img = cv2.imread(item, cv2.IMREAD_COLOR)
            if img is None:
                return np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if isinstance(item, dict):
            if "path" in item and item["path"]:
                img = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
                if img is None:
                    return np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if "bytes" in item and item["bytes"] is not None:
                buf = np.frombuffer(item["bytes"], dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    return np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if "array" in item and item["array"] is not None:
                return self._normalize_np_image(np.asarray(item["array"]))

        try:
            arr = np.asarray(item)  # PIL image -> np array
            return self._normalize_np_image(arr)
        except Exception:
            return np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)

    def _preprocess_rgb_uint8(self, img: np.ndarray) -> torch.Tensor:
        h, w = img.shape[:2]
        scale = RESIZE_SHORT / float(min(h, w) + 1e-6)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = cv2.resize(
            img, (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        )

        h2, w2 = img.shape[:2]
        top = max(0, (h2 - INPUT_SIZE) // 2)
        left = max(0, (w2 - INPUT_SIZE) // 2)
        img = img[top:top + INPUT_SIZE, left:left + INPUT_SIZE, :]

        if img.shape[0] != INPUT_SIZE or img.shape[1] != INPUT_SIZE:
            pad_h = INPUT_SIZE - img.shape[0]
            pad_w = INPUT_SIZE - img.shape[1]
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=0)

        x = torch.from_numpy(img).to(torch.float32) / 255.0
        x = x.permute(2, 0, 1).contiguous()
        x = (x - _MEAN_T) / _STD_T
        return x

    def _load_and_preprocess_one(self, item: Any) -> torch.Tensor:
        img = self._to_rgb_uint8(item)
        return self._preprocess_rgb_uint8(img)

    def _batch_to_tensor(self, batch) -> torch.Tensor:
        if torch.is_tensor(batch):
            return batch

        if isinstance(batch, np.ndarray):
            if batch.ndim == 4 and batch.shape[-1] == 3:
                return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
            if batch.dtype == object or batch.dtype.type is np.str_:
                batch = batch.tolist()

        if isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                return torch.empty(0, 3, INPUT_SIZE, INPUT_SIZE)
            if torch.is_tensor(batch[0]):
                return torch.stack(list(batch), dim=0)
            xs = [self._load_and_preprocess_one(b) for b in batch]
            return torch.stack(xs, dim=0)

        return torch.stack([self._load_and_preprocess_one(batch)], dim=0)

    # -------- model core --------
    def forward_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone_features(x)
        feat = self.gem(feat).flatten(1)
        feat = self.dropout(feat)
        logits = self.cell_head(feat)
        offsets = self.offset_head(feat)
        return logits, offsets

    def decode_latlon(self, logits: torch.Tensor, offsets_raw: torch.Tensor) -> torch.Tensor:
        B, K = logits.shape
        p = F.softmax(logits / self.softmax_temp, dim=-1)

        off = torch.tanh(offsets_raw).view(B, K, 2) * 0.5  # cell units in [-0.5,0.5]
        cand = self.cell_centers_deg.view(1, K, 2) + off * self.cell_size_deg.view(1, 1, 2)
        pred = torch.sum(p.unsqueeze(-1) * cand, dim=1)

        pred_lat = pred[:, 0].clamp(self.lat_min, self.lat_max)
        pred_lon = pred[:, 1].clamp(self.lon_min, self.lon_max)
        return torch.stack([pred_lat, pred_lon], dim=-1)

    def forward(self, batch):
        x = self._batch_to_tensor(batch)
        device = next(self.parameters()).device
        x = x.to(device, non_blocking=True)
        logits, offsets = self.forward_raw(x)
        return self.decode_latlon(logits, offsets)

    def predict(self, batch) -> np.ndarray:
        self.eval()
        with torch.inference_mode():
            pred = self.forward(batch)
        return pred.detach().cpu().numpy().astype(np.float32)

    # -------- training helpers --------
    def encode_cell_and_offset(self, tgt_latlon_deg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lat = tgt_latlon_deg[:, 0]
        lon = tgt_latlon_deg[:, 1]
        dlat = self.cell_size_deg[0]
        dlon = self.cell_size_deg[1]

        i = torch.floor((lat - self.lat_min) / dlat).to(torch.long).clamp(0, self.grid_h - 1)
        j = torch.floor((lon - self.lon_min) / dlon).to(torch.long).clamp(0, self.grid_w - 1)
        cell_id = i * self.grid_w + j

        centers = self.cell_centers_deg[cell_id]
        off_lat = (lat - centers[:, 0]) / (dlat + 1e-12)
        off_lon = (lon - centers[:, 1]) / (dlon + 1e-12)
        off = torch.stack([off_lat, off_lon], dim=-1).clamp(-0.5, 0.5)
        return cell_id, off

    def compute_loss(
        self,
        logits: torch.Tensor,
        offsets_raw: torch.Tensor,
        tgt_latlon_deg: torch.Tensor,
        w_ce: float = 1.0,
        w_off: float = 2.0,
        w_geo: float = 1.0,
        label_smoothing: float = 0.05,
        huber_beta_m: float = 10.0,
        add_mixture_geo: bool = True,
        w_mix_geo: float = 0.3,
    ):
        B, K = logits.shape
        device = logits.device

        cell_id, off_tgt = self.encode_cell_and_offset(tgt_latlon_deg)

        ce = F.cross_entropy(logits, cell_id, label_smoothing=label_smoothing)

        off_all = torch.tanh(offsets_raw).view(B, K, 2) * 0.5
        idx = torch.arange(B, device=device)
        off_pred = off_all[idx, cell_id]
        off = F.smooth_l1_loss(off_pred, off_tgt, beta=0.05)

        # Always used by off-loss
        pred_true = self.cell_centers_deg[cell_id] + off_pred * self.cell_size_deg.view(1, 2)

        loss = (w_ce * ce) + (w_off * off)

        # ---- GEO term: only compute WITH grad if weight != 0 ----
        geo = torch.zeros((), device=device)
        if w_geo != 0.0:
            dist_m = haversine_m(pred_true, tgt_latlon_deg)
            geo = F.smooth_l1_loss(dist_m, torch.zeros_like(dist_m), beta=huber_beta_m)
            loss = loss + (w_geo * geo)
            dist_for_stats = dist_m
        else:
            with torch.no_grad():
                dist_for_stats = haversine_m(pred_true, tgt_latlon_deg)

        # ---- MIX term: only compute WITH grad if weight != 0 AND enabled ----
        mix_geo = torch.zeros((), device=device)
        if add_mixture_geo and (w_mix_geo != 0.0):
            pred_mix = self.decode_latlon(logits, offsets_raw)
            dist_mix = haversine_m(pred_mix, tgt_latlon_deg)
            mix_geo = F.smooth_l1_loss(dist_mix, torch.zeros_like(dist_mix), beta=huber_beta_m)
            loss = loss + (w_mix_geo * mix_geo)

        stats = {
            "loss": float(loss.detach().cpu()),
            "ce": float(ce.detach().cpu()),
            "off": float(off.detach().cpu()),
            "geo": float(geo.detach().cpu()),
            "mix_geo": float(mix_geo.detach().cpu()),
            "dist_m_mean_truecell": float(dist_for_stats.mean().detach().cpu()),
        }
        return loss, stats


def get_model():
    return Model()