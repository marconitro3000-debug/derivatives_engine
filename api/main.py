from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.market_data import router as market_data_router
from api.routes.portfolio import router as portfolio_router
from api.routes.pricing import router as pricing_router
from api.routes.structured import router as structured_router
from api.routes.term_sheet import router as term_sheet_router
from api.routes.validation import router as validation_router
from api.routes.vanilla_option import router as vanilla_option_router
from db.database import init_db

app = FastAPI(title="Derivatives Engine API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "derivatives-engine"}


app.include_router(pricing_router)
app.include_router(validation_router)
app.include_router(structured_router)
app.include_router(term_sheet_router)
app.include_router(market_data_router)
app.include_router(vanilla_option_router)
app.include_router(portfolio_router)
