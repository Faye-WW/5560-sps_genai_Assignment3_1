import io, base64, torch
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from torchvision.utils import save_image
from gan.models import Generator, LATENT_DIM

router = APIRouter(prefix="/gan", tags=["gan"])

# Choose device
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))

# Load generator model
G = Generator().to(DEVICE)
try:
    G.load_state_dict(torch.load("artifacts/generator.pt", map_location=DEVICE))
    G.eval()
except Exception:
    print("⚠️ Warning: generator weights not found or not loaded.")
    G.eval()


@router.get("/health")
def health():
    """Check GAN health status"""
    return {"status": "ok"}


@router.get("/sample")
def sample(n: int = 8):
    """
    Return base64-encoded image (for API testing and JSON use)
    """
    n = max(1, min(n, 16))
    z = torch.randn(n, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        imgs = (G(z).cpu() + 1) / 2
    buf = io.BytesIO()
    save_image(imgs, buf, nrow=n, format="PNG")
    return {
        "image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        "count": n
    }


@router.get("/sample.png")
def sample_png(n: int = Query(8, ge=1, le=16)):
    """
    Return generated digits as PNG image directly viewable in browser.
    This is only for visualization; it does not replace /sample.
    """
    z = torch.randn(n, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        imgs = (G(z).cpu() + 1) / 2
    buf = io.BytesIO()
    save_image(imgs, buf, nrow=n, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@router.get("/preview", response_class=HTMLResponse)
def preview_page(n: int = 8):
    """
    Simple HTML page to preview GAN output directly in browser.
    """
    html = f"""
    <html>
        <head><title>GAN Preview</title></head>
        <body>
            <h3>Generated MNIST Digits (n={n})</h3>
            <img src="/gan/sample.png?n={n}" alt="GAN digits" />
        </body>
    </html>
    """
    return HTMLResponse(content=html)
