# offence_rules.py
"""
This module contains the dictionary of traffic offences, their sections,
typical fines, and advice messages in simple language.

To add a new offence, just add a new dictionary key like this:
"wrong parking": {"offence": "...", "section": "...", "fine": "...", "advice": "..."}
"""

import re

# Clean and extract fine amount from OCR text
def extract_fine_amount(text):
    text_lower = text.lower()
    match = re.search(r"(?:rs\.?|₹|amount[:\s]*)(\d{2,6})", text_lower)
    return match.group(1) if match else "Not mentioned"

# Master offence mapping
OFFENCE_DICT = {
    "helmet": {
        "offence": "Not Wearing Helmet",
        "section": "Section 129, Motor Vehicles Act",
        "fine": "₹500",
        "advice": "Wear a BIS-approved helmet. It protects you and avoids ₹500 fine."
    },
    "signal": {
        "offence": "Signal Jumping (Red Light Violation)",
        "section": "Section 184, Motor Vehicles Act",
        "fine": "₹1000",
        "advice": "Always stop at red lights. Signal jumping is dangerous and fined ₹1000."
    },
    "traffic light": {
        "offence": "Signal Jumping (Traffic Light Violation)",
        "section": "Section 184, Motor Vehicles Act",
        "fine": "₹1000",
        "advice": "Follow traffic signals properly. Pay the fine online or at the traffic office."
    },
    "license": {
        "offence": "Driving Without License",
        "section": "Section 181, Motor Vehicles Act",
        "fine": "₹5000",
        "advice": "Driving without a valid license is punishable. Apply or renew your license."
    },
    "licence": {
        "offence": "Driving Without Licence",
        "section": "Section 181, Motor Vehicles Act",
        "fine": "₹5000",
        "advice": "Driving without a valid licence attracts ₹5000 fine. Carry it always."
    },
    "speed": {
        "offence": "Overspeeding",
        "section": "Section 183, Motor Vehicles Act",
        "fine": "₹1000 (two-wheeler), ₹2000 (LMV)",
        "advice": "Follow the speed limit. Overspeeding causes accidents and fines."
    },
    "overspeed": {
        "offence": "Overspeeding",
        "section": "Section 183, Motor Vehicles Act",
        "fine": "₹1000 (two-wheeler), ₹2000 (LMV)",
        "advice": "Avoid speeding. It can lead to heavy fines and suspension of license."
    },
    "seatbelt": {
        "offence": "Not Wearing Seatbelt",
        "section": "Section 194B, Motor Vehicles Act",
        "fine": "₹1000",
        "advice": "Always wear a seatbelt for safety and to avoid fines."
    },
    "mobile": {
        "offence": "Using Mobile While Driving",
        "section": "Section 184, Motor Vehicles Act",
        "fine": "₹1000–₹5000",
        "advice": "Avoid mobile usage while driving. It's risky and fined heavily."
    },
    "phone": {
        "offence": "Using Mobile While Driving",
        "section": "Section 184, Motor Vehicles Act",
        "fine": "₹1000–₹5000",
        "advice": "Don't use your phone while driving. Pull over safely if needed."
    },
    "triple": {
        "offence": "Triple Riding on Two-Wheeler",
        "section": "Section 128, Motor Vehicles Act",
        "fine": "₹1000",
        "advice": "Only one pillion rider is allowed. Avoid triple riding."
    },
    "drunk": {
        "offence": "Drunk Driving",
        "section": "Section 185, Motor Vehicles Act",
        "fine": "₹10,000 and/or imprisonment",
        "advice": "Never drive under alcohol influence. It endangers lives and invites arrest."
    },
    "alcohol": {
        "offence": "Drunk Driving",
        "section": "Section 185, Motor Vehicles Act",
        "fine": "₹10,000 and/or imprisonment",
        "advice": "Driving under influence is illegal and punishable."
    },
    "parking": {
        "offence": "Wrong Parking",
        "section": "Rule 15, Central Motor Vehicles Rules",
        "fine": "₹500",
        "advice": "Park only in designated areas. Wrong parking causes obstruction."
    },
    "pollution": {
        "offence": "No Pollution Certificate",
        "section": "Section 190(2), Motor Vehicles Act",
        "fine": "₹10,000",
        "advice": "Keep a valid PUC certificate to avoid fines."
    },
    "insurance": {
        "offence": "No Insurance",
        "section": "Section 146, Motor Vehicles Act",
        "fine": "₹2000 (first), ₹4000 (repeat)",
        "advice": "Always renew your vehicle insurance. It's mandatory."
    },
    "rc": {
        "offence": "No Registration Certificate",
        "section": "Section 39, Motor Vehicles Act",
        "fine": "₹5000",
        "advice": "Keep your RC available at all times."
    },
    "wrong side": {
        "offence": "Wrong Side Driving",
        "section": "Section 119/177, Motor Vehicles Act",
        "fine": "₹500–₹1000",
        "advice": "Follow lane rules and direction signs."
    },
    "number plate": {
        "offence": "Driving Without Number Plate",
        "section": "Section 39, Motor Vehicles Act",
        "fine": "₹5000",
        "advice": "Install proper number plates as per the standard."
    },
    "dangerous": {
        "offence": "Dangerous Driving",
        "section": "Section 184, Motor Vehicles Act",
        "fine": "₹5000 and/or imprisonment",
        "advice": "Drive safely. Rash driving causes accidents and strict penalties."
    },
}

# --------------------------------------------------------------------
# Match a detected offence
# --------------------------------------------------------------------
def detect_offence(text):
    """Find offence by keyword match and return structured info."""
    text_lower = text.lower()
    text_lower = text_lower.replace("\n", " ").replace("  ", " ")

    for key, data in OFFENCE_DICT.items():
        if key in text_lower:
            return data

    # Nothing matched
    return {
        "offence": "Unknown / Not Detected",
        "section": "-",
        "fine": "Not mentioned",
        "advice": "Could not identify the offence. Please check or upload a clearer challan image."
    }
