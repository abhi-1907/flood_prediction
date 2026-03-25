"""
Alert Composer – Generates alert content using LLM and templates.

Produces multi-format alert content:
  - Plain text (SMS)
  - HTML (email)
  - Push notification (title + body)
  - Webhook payload (JSON)

Content is tailored to:
  - Alert severity (GREEN/YELLOW/ORANGE/RED)
  - User type (public vs authority)
  - Location-specific details
  - Prediction and recommendation data
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agents.alerting.alerting_schemas import (
    AlertContent,
    AlertSeverity,
)
from services.gemini_service import GeminiService
from utils.logger import logger


# ── Severity configuration ────────────────────────────────────────────────────

SEVERITY_EMOJI = {
    AlertSeverity.GREEN:  "🟢",
    AlertSeverity.YELLOW: "🟡",
    AlertSeverity.ORANGE: "🟠",
    AlertSeverity.RED:    "🔴",
}

SEVERITY_LABEL = {
    AlertSeverity.GREEN:  "ALL CLEAR",
    AlertSeverity.YELLOW: "FLOOD WATCH",
    AlertSeverity.ORANGE: "FLOOD WARNING",
    AlertSeverity.RED:    "FLOOD EMERGENCY",
}


class AlertComposer:
    """
    Generates multi-format alert content.

    Usage:
        composer = AlertComposer(gemini_service)
        content  = await composer.compose(severity, risk_data, location_data, for_authority)
    """

    def __init__(self, gemini_service: Optional[GeminiService] = None) -> None:
        self._gemini = gemini_service

    async def compose(
        self,
        severity:        AlertSeverity,
        risk_data:       Dict[str, Any],
        location_data:   Dict[str, Any],
        for_authority:   bool = False,
        recommendations: Optional[List[Dict[str, Any]]] = None,
    ) -> AlertContent:
        """
        Generates complete alert content for all channels.

        Args:
            severity:        Alert severity level.
            risk_data:       Prediction results (flood_probability, risk_level, etc.).
            location_data:   Location info (name, state, emergency_number, etc.).
            for_authority:   Whether this is for government officials.
            recommendations: Optional list of recommendation dicts from RecommendationAgent.

        Returns:
            AlertContent with text, HTML, push, and webhook formats.
        """
        location   = location_data.get("location", "your area")
        flood_prob = risk_data.get("flood_probability", 0.5)
        risk_level = risk_data.get("risk_level", "MEDIUM")
        confidence = risk_data.get("confidence", 0.5)
        emergency  = location_data.get("emergency_number", "112")
        emoji      = SEVERITY_EMOJI.get(severity, "⚠️")
        label      = SEVERITY_LABEL.get(severity, "ALERT")

        # ── 1. SMS / plain text ───────────────────────────────────────────
        body_text = self._compose_sms(
            emoji, label, location, flood_prob, risk_level, emergency, for_authority
        )

        # ── 2. Push notification ──────────────────────────────────────────
        push_title = f"{emoji} {label}: {location}"
        push_body  = f"Flood probability {flood_prob*100:.0f}% ({risk_level}). Emergency: {emergency}"

        # ── 3. HTML email ─────────────────────────────────────────────────
        body_html = await self._compose_html(
            severity, emoji, label, location, flood_prob,
            risk_level, confidence, emergency, risk_data,
            location_data, for_authority,
            recommendations=recommendations or [],
        )

        # ── 4. Webhook payload ────────────────────────────────────────────
        webhook_payload = {
            "event":            "flood_alert",
            "severity":         severity.value,
            "risk_level":       risk_level,
            "flood_probability": flood_prob,
            "confidence":       confidence,
            "location":         location,
            "latitude":         location_data.get("latitude"),
            "longitude":        location_data.get("longitude"),
            "emergency_number": emergency,
            "message":          body_text,
            "for_authority":    for_authority,
        }

        subject = f"{emoji} {label} — {location} ({flood_prob*100:.0f}% probability)"

        logger.info(
            f"[AlertComposer] Composed {severity.value} alert for {location} | "
            f"authority={for_authority}"
        )

        return AlertContent(
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            push_title=push_title,
            push_body=push_body,
            webhook_payload=webhook_payload,
            metadata={
                "severity": severity.value,
                "risk_level": risk_level,
                "flood_probability": flood_prob,
                "location": location,
            },
        )

    # ── SMS composer ──────────────────────────────────────────────────────

    @staticmethod
    def _compose_sms(
        emoji:        str,
        label:        str,
        location:     str,
        flood_prob:   float,
        risk_level:   str,
        emergency:    str,
        for_authority: bool,
    ) -> str:
        """Generates a short SMS message (< 300 chars)."""
        if for_authority:
            return (
                f"{emoji} {label} — {location}\n"
                f"Flood probability: {flood_prob*100:.0f}% ({risk_level})\n"
                f"Action: Activate emergency protocols. "
                f"Deploy rescue teams. Alert downstream villages.\n"
                f"Contact: {emergency}"
            )[:300]

        if risk_level in ("HIGH", "CRITICAL"):
            return (
                f"{emoji} {label} for {location}!\n"
                f"Flood probability: {flood_prob*100:.0f}%\n"
                f"Move to higher ground IMMEDIATELY.\n"
                f"Pack essential documents and medicines.\n"
                f"Emergency: {emergency}"
            )[:300]

        return (
            f"{emoji} {label} for {location}.\n"
            f"Flood probability: {flood_prob*100:.0f}%.\n"
            f"Stay alert and prepare essential supplies.\n"
            f"Emergency: {emergency}"
        )[:300]

    # ── HTML email composer ───────────────────────────────────────────────

    async def _compose_html(
        self,
        severity:        AlertSeverity,
        emoji:           str,
        label:           str,
        location:        str,
        flood_prob:      float,
        risk_level:      str,
        confidence:      float,
        emergency:       str,
        risk_data:       Dict[str, Any],
        location_data:   Dict[str, Any],
        for_authority:   bool,
        recommendations: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generates HTML email content with styling."""
        # Color scheme based on severity
        bg_color = {
            AlertSeverity.GREEN:  "#E8F5E9",
            AlertSeverity.YELLOW: "#FFF8E1",
            AlertSeverity.ORANGE: "#FFF3E0",
            AlertSeverity.RED:    "#FFEBEE",
        }.get(severity, "#FFF8E1")

        accent_color = {
            AlertSeverity.GREEN:  "#2E7D32",
            AlertSeverity.YELLOW: "#F9A825",
            AlertSeverity.ORANGE: "#E65100",
            AlertSeverity.RED:    "#C62828",
        }.get(severity, "#E65100")

        # LLM-generated advisory paragraph
        advisory = await self._llm_advisory(
            severity, location, flood_prob, risk_level, for_authority
        )

        # Key factors from prediction
        key_factors = risk_data.get("key_factors", [])
        factors_html = ""
        if key_factors:
            factors_html = "<ul>" + "".join(
                f"<li>{f}</li>" for f in key_factors[:4]
            ) + "</ul>"

        # Authority-specific section
        auth_section = ""
        if for_authority:
            auth_section = f"""
            <div style="background:#f5f5f5;padding:12px;border-radius:6px;margin:12px 0;">
                <h3 style="margin:0 0 8px 0;">🏛️ Official Action Required</h3>
                <ol>
                    <li>Activate District Emergency Operations Centre</li>
                    <li>Alert NDRF / SDRF teams for deployment</li>
                    <li>Issue public warning via loudspeakers and media</li>
                    <li>Open emergency shelters and relief camps</li>
                    <li>Coordinate with police for traffic and evacuation</li>
                </ol>
            </div>"""

        # Recommendations section
        recs_html = ""
        if recommendations:
            urgency_colors = {
                "emergency":     "#C62828",
                "warning":       "#E65100",
                "advisory":      "#F9A825",
                "informational": "#1565C0",
            }
            rec_items = ""
            for r in (recommendations or [])[:5]:   # Cap at 5
                urgency    = r.get("urgency", "advisory")
                color      = urgency_colors.get(urgency, "#555")
                title      = r.get("title", "")
                title_reg  = r.get("title_regional")
                desc       = r.get("description", "")
                desc_reg   = r.get("description_regional")
                steps      = r.get("action_steps", [])
                steps_reg  = r.get("action_steps_regional", [])
                
                if title_reg or desc_reg or (steps_reg and len(steps_reg) > 0):
                    logger.info(f"[AlertComposer] Rendering regional content for '{title}': title_reg={'yes' if title_reg else 'no'}, desc_reg={'yes' if desc_reg else 'no'}, steps_reg={len(steps_reg) if steps_reg else 0}")
                else:
                    logger.debug(f"[AlertComposer] No regional content for '{title}'")

                # Combine English and Regional steps
                steps_html = "".join(f"<li style='margin:2px 0'>{s}</li>" for s in (steps or [])[:3])
                steps_reg_html = "".join(f"<li style='margin:2px 0; font-style: italic;'>{s}</li>" for s in (steps_reg or [])[:3])

                rec_items += (
                    f'<div style="border-left:4px solid {color};padding:8px 12px;'
                    f'margin:8px 0;background:#fafafa;border-radius:0 6px 6px 0;">'
                    f'<strong style="color:{color};">[{urgency.upper()}]</strong> {title}'
                    + (f' | <span style="font-weight:normal;">{title_reg}</span>' if title_reg else "")
                    + f'<br><span style="font-size:13px;color:#555;">{desc}</span>'
                    + (f'<br><span style="font-size:13px;color:#666;font-style:italic;">{desc_reg}</span>' if desc_reg else "")
                    + (f'<ul style="margin:4px 0;padding-left:16px;font-size:13px;color:#444;">'
                       f'{steps_html}{steps_reg_html}</ul>' if steps_html or steps_reg_html else "")
                    + "</div>"
                )
            recs_html = (
                f'<div style="border-top:1px solid #eee;padding-top:12px;margin-top:12px;">'
                f'<h3 style="color:{accent_color};margin:0 0 8px 0;">📋 Safety Recommendations</h3>'
                f'{rec_items}'
                f'</div>'
            )

        html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family:'Segoe UI',Roboto,sans-serif;margin:0;padding:0;background:#f0f0f0;">
<div style="max-width:600px;margin:20px auto;background:white;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">

    <!-- Header -->
    <div style="background:{accent_color};padding:20px;text-align:center;color:white;">
        <h1 style="margin:0;font-size:24px;">{emoji} {label}</h1>
        <p style="margin:8px 0 0 0;font-size:16px;opacity:0.9;">{location}</p>
    </div>

    <!-- Risk meter -->
    <div style="background:{bg_color};padding:16px;text-align:center;">
        <div style="font-size:48px;font-weight:bold;color:{accent_color};">
            {flood_prob*100:.0f}%
        </div>
        <div style="font-size:14px;color:#666;">Flood Probability | Confidence: {confidence*100:.0f}%</div>
    </div>

    <!-- Advisory -->
    <div style="padding:20px;">
        <p style="font-size:15px;line-height:1.6;color:#333;">{advisory}</p>

        {f'<h3 style="color:{accent_color};">Key Risk Factors</h3>{factors_html}' if factors_html else ''}

        {auth_section}

        {recs_html}

        <!-- Emergency contact -->
        <div style="background:#E3F2FD;padding:12px;border-radius:6px;margin:16px 0;text-align:center;">
            <strong>📞 Emergency Helpline: {emergency}</strong><br>
            <small>NDRF: 011-24363260 | Police: 100 | Ambulance: 108</small>
        </div>

        <!-- Safety tips -->
        <div style="border-top:1px solid #eee;padding-top:12px;margin-top:12px;">
            <h3 style="color:{accent_color};margin:0 0 8px 0;">🛡️ General Safety Tips</h3>
            <ul style="color:#555;font-size:14px;">
                <li>Keep phone charged and save emergency contacts</li>
                <li>Store clean drinking water and essential medicines</li>
                <li>Do NOT walk or drive through floodwater</li>
                <li>Move to higher ground if water begins rising</li>
            </ul>
        </div>
    </div>

    <!-- Footer -->
    <div style="background:#f5f5f5;padding:12px;text-align:center;font-size:12px;color:#999;">
        FloodSense AI | Automated Alert System<br>
        This is an automated alert. Do not reply to this email.
    </div>
</div>
</body>
</html>"""
        return html

    # ── LLM advisory ──────────────────────────────────────────────────────

    async def _llm_advisory(
        self,
        severity:      AlertSeverity,
        location:      str,
        flood_prob:    float,
        risk_level:    str,
        for_authority: bool,
    ) -> str:
        """Generates a 2–3 sentence LLM advisory for the alert body."""
        if not self._gemini:
            return self._fallback_advisory(severity, location, flood_prob, for_authority)

        try:
            audience = "government officials" if for_authority else "the general public"
            prompt = f"""
Write a 2-sentence flood alert advisory for {audience}:
- Location: {location}
- Severity: {severity.value}
- Risk level: {risk_level}
- Probability: {flood_prob*100:.0f}%

Be direct, urgent (if HIGH/CRITICAL), and actionable. No headers.
"""
            result = await self._gemini.generate(prompt, use_fast_model=True)
            return result.strip() if result else self._fallback_advisory(
                severity, location, flood_prob, for_authority
            )
        except Exception:
            return self._fallback_advisory(severity, location, flood_prob, for_authority)

    @staticmethod
    def _fallback_advisory(
        severity:      AlertSeverity,
        location:      str,
        flood_prob:    float,
        for_authority: bool,
    ) -> str:
        if severity in (AlertSeverity.RED, AlertSeverity.ORANGE):
            if for_authority:
                return (
                    f"Flood probability in {location} has reached {flood_prob*100:.0f}%. "
                    "Activate emergency response protocols and deploy rescue teams immediately."
                )
            return (
                f"Flood risk in {location} is dangerously high ({flood_prob*100:.0f}%). "
                "Move to higher ground immediately and follow instructions from local authorities."
            )
        return (
            f"Flood probability in {location} is {flood_prob*100:.0f}%. "
            "Stay alert and monitor local weather updates. Prepare essential supplies."
        )
