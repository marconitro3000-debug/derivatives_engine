from __future__ import annotations

import requests
import os


class SecEdgarSource:
    name = "sec_edgar"
    submissions_base = "https://data.sec.gov/submissions"
    companyfacts_base = "https://data.sec.gov/api/xbrl/companyfacts"

    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT", "DerivativesEngine/0.1 research@example.com")

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}

    @staticmethod
    def cik10(cik: int | str) -> str:
        return str(cik).lstrip("CIK").zfill(10)

    def submissions(self, cik: int | str) -> dict:
        url = f"{self.submissions_base}/CIK{self.cik10(cik)}.json"
        r = requests.get(url, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def companyfacts(self, cik: int | str) -> dict:
        url = f"{self.companyfacts_base}/CIK{self.cik10(cik)}.json"
        r = requests.get(url, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()
