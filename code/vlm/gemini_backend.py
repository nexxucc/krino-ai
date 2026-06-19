"""
Google Gemini vision backend (default; free tier) via the REST API.

Transport is plain `requests` (no SDK dependency). Each claim is ONE multimodal
call: the prompt plus every usable image as inline base64 JPEG. Output is forced
to JSON (`responseMimeType=application/json`) at temperature 0 for determinism.

Built-in reliability for the free tier:
  - client-side throttle to stay under RPM,
  - exponential backoff + jitter on 429/503 (honors Retry-After),
  - after max retries the caller (run.py) falls back to the heuristic backend.

Secrets: the API key is read from config (env / .env) and sent in the
`x-goog-api-key` header — never placed in the URL or logged.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import List, Optional

import requests

from claim_parser import ParsedClaim
from image_utils import ImageSet
from vlm.base import VLMBackend, VLMObservation
from vlm.prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    VERIFIER_SYSTEM,
    build_user_prompt,
    build_verifier_prompt,
)


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    brace = re.search(r"\{.*\}", text, re.S)
    for c in [fence.group(1) if fence else None, brace.group(0) if brace else None]:
        if c:
            try:
                return json.loads(c)
            except json.JSONDecodeError:
                continue
    return None


class _RateLimiter:
    """Simple sequential min-interval limiter (requests are made one at a time)."""

    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(1, rpm)
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


class GeminiBackend(VLMBackend):
    name = "gemini"

    def __init__(self, api_key: str, model: str, endpoint: str, timeout: int = 120,
                 temperature: float = 0.0, rpm: int = 8, max_retries: int = 3,
                 max_images: int = 3, thinking_budget: int = 512,
                 max_output_tokens: int = 2048):
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_images = max_images
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.limiter = _RateLimiter(rpm)
        # circuit breaker: after this many calls fully exhaust their retries we
        # assume the daily quota is dead and fail fast (stop wasting requests).
        self._hard_fails = 0
        self._fail_fast_after = 2
        # label the variant so the eval report / cache key reflects the model,
        # the prompt version, AND the thinking budget (changing any of these
        # invalidates the cache so observations stay internally consistent).
        self.name = f"gemini:{model}:{PROMPT_VERSION}:t{thinking_budget}"

    def available(self) -> bool:
        return bool(self.api_key)

    def _url(self) -> str:
        return f"{self.endpoint}/models/{self.model}:generateContent"

    @staticmethod
    def _retry_delay_seconds(resp) -> Optional[float]:
        """Pull Google's suggested retry delay (RetryInfo.retryDelay, e.g. '29s')."""
        try:
            for d in resp.json().get("error", {}).get("details", []):
                if "RetryInfo" in d.get("@type", "") and d.get("retryDelay"):
                    return float(str(d["retryDelay"]).rstrip("s"))
        except (ValueError, KeyError, TypeError):
            pass
        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                return float(ra)
            except ValueError:
                return None
        return None

    def _post_with_retry(self, payload: dict) -> dict:
        # Circuit breaker: if earlier calls already died on quota, stop trying —
        # the daily cap is gone and further requests are wasted.
        if self._hard_fails >= self._fail_fast_after:
            raise RuntimeError("gemini quota appears exhausted; failing fast")

        last_exc = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                resp = requests.post(
                    self._url(),
                    headers={"x-goog-api-key": self.api_key,
                             "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                self._backoff(attempt, None)
                continue
            if resp.status_code == 200:
                self._hard_fails = 0  # healthy response resets the breaker
                return resp.json()
            if resp.status_code == 429:
                # Could be per-MINUTE (recoverable) or per-DAY (not). We cannot
                # tell from the message, so retry a couple of times honoring the
                # server's retryDelay; the circuit breaker handles a dead key.
                last_exc = requests.HTTPError(f"429: {resp.text[:160]}")
                if attempt < self.max_retries - 1:
                    delay = self._retry_delay_seconds(resp)
                    time.sleep(min(delay if delay else (5 * (attempt + 1)), 40.0))
                    continue
                self._hard_fails += 1
                raise RuntimeError(f"gemini 429 after {self.max_retries} tries: {last_exc}")
            if resp.status_code in (500, 503):
                last_exc = requests.HTTPError(f"{resp.status_code}: {resp.text[:160]}")
                self._backoff(attempt, resp.headers.get("Retry-After"))
                continue
            # non-retryable (400/401/403) — surface immediately
            resp.raise_for_status()
        raise RuntimeError(f"gemini call failed after {self.max_retries} retries: {last_exc}")

    @staticmethod
    def _backoff(attempt: int, retry_after: Optional[str]) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 60.0))
                return
            except (ValueError, TypeError):
                pass
        # exponential backoff with jitter: 1,2,4,8,16s (+/- jitter)
        delay = min(2 ** attempt, 16) + random.uniform(0, 1.0)
        time.sleep(delay)

    def analyze(self, image_set, parsed, claim_object, user_claim, requirements_text):
        usable = image_set.usable[: self.max_images]
        obs = VLMObservation()
        if not usable:
            obs.object_match = False
            obs.part_visible = False
            obs.notes = "No usable image to send to the model."
            return obs

        user_prompt = build_user_prompt(
            claim_object=claim_object,
            claimed_part=parsed.claimed_part,
            claimed_issue=parsed.issue_type,
            user_claim=user_claim,
            requirements_text=requirements_text,
            image_ids=[im.image_id for im in usable],
        )
        parts = [{"text": user_prompt}]
        for im in usable:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": im.b64()}})

        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": parts}],
            # thinking budget shares maxOutputTokens; both set in _gen_config so the
            # response (thinking + JSON) is never truncated and stays quota-safe.
            "generationConfig": self._gen_config(),
        }
        data = self._post_with_retry(payload)
        text = self._first_text(data)
        parsed_json = _extract_json(text)
        if parsed_json is None:
            raise ValueError("gemini returned unparseable output")
        return self._to_observation(parsed_json, usable, parsed)

    def _gen_config(self) -> dict:
        return {
            "temperature": self.temperature,
            "responseMimeType": "application/json",
            "maxOutputTokens": self.max_output_tokens,
            "thinkingConfig": {"thinkingBudget": self.thinking_budget},
        }

    def challenge(self, image_set, parsed, claim_object, user_claim, obs):
        """Adversarial second opinion: re-examine the images and possibly correct
        the first-pass verdict. Returns a (possibly) updated VLMObservation. On
        any failure the original observation is returned unchanged."""
        import dataclasses

        from vlm.prompt import build_verifier_prompt

        usable = image_set.usable[: self.max_images]
        if not usable:
            return obs
        prompt = build_verifier_prompt(
            claim_object=claim_object,
            claimed_part=parsed.claimed_part,
            claimed_issue=parsed.issue_type,
            user_claim=user_claim,
            first_status=obs.proposed_status,
            first_issue=obs.visible_issue_type,
            first_severity=obs.severity,
            image_ids=[im.image_id for im in usable],
        )
        parts = [{"text": prompt}]
        for im in usable:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": im.b64()}})
        payload = {
            "system_instruction": {"parts": [{"text": VERIFIER_SYSTEM}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": self._gen_config(),
        }
        try:
            data = self._post_with_retry(payload)
            vj = _extract_json(self._first_text(data))
        except Exception:
            return obs
        if vj is None:
            return obs

        new_status = str(vj.get("corrected_status", obs.proposed_status))
        new_issue = str(vj.get("corrected_issue_type", obs.visible_issue_type))
        new_sev = str(vj.get("corrected_severity", obs.severity))
        new_dmg = bool(vj.get("corrected_damage_present", obs.damage_present))
        try:
            vconf = max(0.0, min(1.0, float(vj.get("confidence", obs.confidence))))
        except (ValueError, TypeError):
            vconf = obs.confidence
        note = str(vj.get("reason", ""))[:200]
        return dataclasses.replace(
            obs,
            proposed_status=new_status,
            visible_issue_type=new_issue if new_issue else obs.visible_issue_type,
            severity=new_sev if new_sev else obs.severity,
            damage_present=new_dmg,
            confidence=vconf,
            notes=(note or obs.notes),
        )

    @staticmethod
    def _first_text(data: dict) -> str:
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def _to_observation(data: dict, usable, parsed: ParsedClaim) -> VLMObservation:
        valid_ids = {im.image_id for im in usable}
        raw_ids = data.get("supporting_image_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        ids = [str(i).strip() for i in raw_ids if str(i).strip() in valid_ids]

        try:
            conf = float(data.get("confidence", 0.0))
        except (ValueError, TypeError):
            conf = 0.0

        obs = VLMObservation(
            object_match=bool(data.get("object_match", True)),
            part_visible=bool(data.get("part_visible", False)),
            visible_issue_type=str(data.get("visible_issue_type", "unknown")),
            visible_object_part=str(data.get("visible_object_part", "unknown")),
            damage_present=bool(data.get("damage_present", False)),
            severity=str(data.get("severity", "unknown")),
            proposed_status=str(data.get("proposed_status", "not_enough_information")),
            confidence=max(0.0, min(1.0, conf)),
            supporting_image_ids=ids,
            quality_flags=[str(q) for q in (data.get("quality_issues") or [])],
            possible_manipulation=bool(data.get("possible_manipulation", False)),
            non_original_image=bool(data.get("non_original_image", False)),
            text_instruction_in_image=bool(data.get("text_instruction_in_image", False)),
            notes=str(data.get("notes", ""))[:300],
            raw=data,
        )
        obs.text_instruction_in_image = obs.text_instruction_in_image or parsed.injection_detected
        return obs
