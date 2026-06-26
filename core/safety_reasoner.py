"""Safety reasoning engine for contextual hazard analysis.

Takes raw outputs from image analysis (captions and VQA answers)
and performs rule-based reasoning to identify specific hazards,
assess severity, and generate actionable safety warnings.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from config import HAZARD_PATTERNS, SEVERITY_CONFIG


@dataclass
class HazardWarning:
    """A single identified hazard."""
    hazard_type: str
    hazard_name: str
    severity: str
    icon: str
    confidence: float  # 0.0 - 1.0
    matched_keywords: list = field(default_factory=list)
    dynamic_action: str = ""


@dataclass
class SafetyReport:
    """Complete safety analysis report."""
    primary_hazard: Optional[HazardWarning] = None
    additional_hazards: List[HazardWarning] = field(default_factory=list)
    scene_elements: List[str] = field(default_factory=list)
    explanation: str = ""
    recommended_actions: List[str] = field(default_factory=list)
    raw_analysis: dict = field(default_factory=dict)


class SafetyReasoner:
    """Rule-based safety reasoning engine.
    
    Analyzes text outputs from image analysis to identify
    hazard patterns and generate structured safety warnings.
    """

    # Extended keyword synonyms for better matching
    KEYWORD_EXPANSIONS = {
        "cord": ["cord", "cable", "wire", "extension", "power cord", "electric cord"],
        "water": ["water", "puddle", "wet", "liquid", "pool", "flood", "moisture", "damp"],
        "damaged": ["frayed", "damaged", "broken", "exposed", "worn", "torn", "bare", "old", "deteriorated"],
        "electrical": ["electric", "electrical", "power", "voltage", "current", "plug", "outlet", "socket"],
        "floor": ["floor", "ground", "concrete", "surface", "tile", "pavement"],
    }

    # Hazard explanation templates
    EXPLANATIONS = {
        "electrocution": (
            "**⚡ CRITICAL: Electrocution Hazard Detected**\n\n"
            "The combination of {electrical_elements} in close proximity to {water_elements} "
            "creates an **extremely dangerous electrocution risk**. Water is a conductor of "
            "electricity, and any contact between the energized electrical components and the "
            "water could create a lethal path for electrical current.\n\n"
            "**Why this is particularly dangerous:**\n"
            "- Water conducts electricity and can energize the entire puddle area\n"
            "- Anyone stepping in or near the water could receive a fatal electric shock\n"
            "- {damage_detail}\n"
            "- The concrete floor, when wet, becomes an excellent conductor\n\n"
            "**Potential consequences:** Electric shock, cardiac arrest, burns, or death."
        ),
        "electrical_fire": (
            "**🔥 HIGH RISK: Electrical Fire Hazard Detected**\n\n"
            "The {damage_elements} on the electrical components create a significant fire risk. "
            "Damaged insulation can cause short circuits, arcing, and overheating that may "
            "ignite nearby materials.\n\n"
            "**Potential consequences:** Electrical fire, property damage, smoke inhalation."
        ),
        "electric_shock": (
            "**⚠️ HIGH RISK: Electric Shock Hazard**\n\n"
            "Exposed electrical components pose a direct contact shock risk to anyone "
            "handling or touching the equipment.\n\n"
            "**Potential consequences:** Electric shock, burns, muscle contractions."
        ),
        "slip_fall": (
            "**🚶 MEDIUM RISK: Slip and Fall Hazard**\n\n"
            "Water or liquid on the floor surface creates a slip hazard for anyone "
            "walking through the area.\n\n"
            "**Potential consequences:** Falls, bruises, fractures, head injuries."
        ),
        "trip_hazard": (
            "**🦶 LOW RISK: Trip Hazard**\n\n"
            "Cords or cables on the floor create a tripping hazard.\n\n"
            "**Potential consequences:** Falls, sprains, bruises."
        ),
    }

    # Recommended actions per hazard type
    RECOMMENDED_ACTIONS = {
        "electrocution": [
            "🚫 **DO NOT** touch the water or the electrical cord",
            "⚡ Immediately turn off the power at the circuit breaker or fuse box",
            "📞 Call emergency services (911) if anyone has been shocked",
            "🚧 Evacuate the immediate area and establish a safety perimeter",
            "⚠️ Place warning signs to prevent others from entering the area",
            "🔧 Contact a licensed electrician to inspect and repair the wiring",
            "🧹 Only clean up the water AFTER power has been confirmed off",
        ],
        "electrical_fire": [
            "⚡ Disconnect the power source immediately",
            "🧯 Keep a Class C fire extinguisher nearby",
            "🚫 Do NOT use water to extinguish an electrical fire",
            "📞 Call emergency services if a fire starts",
            "🔧 Replace the damaged cord immediately",
        ],
        "electric_shock": [
            "🚫 Do NOT touch exposed wires with bare hands",
            "⚡ Disconnect the power source",
            "🔧 Replace or repair damaged insulation",
            "🧤 Use insulated tools if handling is necessary",
        ],
        "slip_fall": [
            "⚠️ Place wet floor warning signs",
            "🧹 Clean up the liquid promptly",
            "🚧 Block off the wet area until dry",
        ],
        "trip_hazard": [
            "📦 Secure cords with cable covers or tape",
            "🔄 Reroute cables away from walkways",
        ],
    }

    def analyze(self, caption: str, vqa_answers: dict, raw_texts: list = None) -> SafetyReport:
        """Perform safety analysis on image analysis outputs.
        
        Args:
            caption: The image caption from BLIP.
            vqa_answers: Dictionary of question -> answer from VQA.
            raw_texts: All raw text outputs for keyword extraction.
            
        Returns:
            SafetyReport with identified hazards and recommendations.
        """
        report = SafetyReport()
        
        # Combine all text for keyword analysis
        all_text = self._collect_text(caption, vqa_answers, raw_texts)
        all_text_lower = all_text.lower()
        
        # Extract scene elements
        report.scene_elements = self._extract_scene_elements(all_text_lower)
        report.raw_analysis = {
            "caption": caption,
            "vqa_answers": vqa_answers,
            "detected_keywords": report.scene_elements,
        }
        
        # Check each hazard pattern
        hazards = []
        for hazard_id, pattern_config in HAZARD_PATTERNS.items():
            warning = self._check_hazard_pattern(
                hazard_id, pattern_config, all_text_lower
            )
            if warning:
                hazards.append(warning)
        
        # Run generic safety checks dynamically
        dynamic_warning = self._check_dynamic_hazard(caption, vqa_answers)
        if dynamic_warning:
            hazards.append(dynamic_warning)
        
        # Sort by severity priority (highest first)
        hazards.sort(
            key=lambda h: SEVERITY_CONFIG.get(h.severity, {}).get("priority", 0),
            reverse=True,
        )
        
        # Assign primary and additional hazards
        if hazards:
            report.primary_hazard = hazards[0]
            report.additional_hazards = hazards[1:]
            
            # Generate explanation
            report.explanation = self._generate_explanation(
                report.primary_hazard, all_text_lower
            )
            
            # Get recommended actions
            if report.primary_hazard.hazard_type == "dynamic_hazard":
                rec_actions = []
                if report.primary_hazard.dynamic_action:
                    rec_actions.append(f"🚨 {report.primary_hazard.dynamic_action}")
                rec_actions.extend([
                    "⚠️ Keep a safe distance from the identified danger",
                    "🚧 Warn others in the vicinity about the hazard",
                    "🔧 Contact a safety supervisor or professional technician",
                ])
                report.recommended_actions = rec_actions
            else:
                report.recommended_actions = self.RECOMMENDED_ACTIONS.get(
                    report.primary_hazard.hazard_type, []
                )
        else:
            report.explanation = (
                "No specific safety hazards were identified with high confidence. "
                "However, always exercise caution in unfamiliar environments."
            )
            report.recommended_actions = [
                "Assess the environment for potential hazards",
                "Follow standard safety protocols",
            ]
        
        return report

    def _collect_text(self, caption: str, vqa_answers: dict, raw_texts: list = None) -> str:
        """Combine all text sources into a single string for analysis."""
        parts = [caption]
        for q, a in vqa_answers.items():
            parts.append(a)
        if raw_texts:
            # Filter out any strings that contain the diagnostic questions
            # or just add the items if they are clean answers.
            # To be safe, if we change raw_texts in image_analyzer.py to not contain questions,
            # we can just add them here.
            parts.extend(raw_texts)
        return " ".join(parts)

    def _extract_scene_elements(self, text: str) -> list:
        """Extract identified scene elements from the text."""
        elements = []
        element_keywords = {
            "Electrical cord/cable": ["cord", "cable", "wire", "power cord", "extension"],
            "Water/puddle": ["water", "puddle", "wet", "liquid", "pool"],
            "Damage/fraying": ["frayed", "damaged", "broken", "exposed", "worn", "torn"],
            "Concrete floor": ["concrete", "floor", "ground", "cement"],
            "Electrical components": ["electric", "electrical", "power", "plug", "outlet"],
        }
        for element, keywords in element_keywords.items():
            if any(kw in text for kw in keywords):
                elements.append(element)
        return elements

    def _check_hazard_pattern(self, hazard_id: str, config: dict, text: str) -> Optional[HazardWarning]:
        """Check if a hazard pattern matches the analyzed text."""
        for trigger_groups in config["triggers"]:
            group_a, group_b = trigger_groups
            
            # Find matches in each group
            matches_a = [kw for kw in group_a if kw in text]
            matches_b = [kw for kw in group_b if kw in text]
            
            if matches_a and matches_b:
                # Calculate confidence based on number of keyword matches
                total_possible = len(group_a) + len(group_b)
                total_matched = len(matches_a) + len(matches_b)
                confidence = min(1.0, total_matched / (total_possible * 0.4))
                
                return HazardWarning(
                    hazard_type=hazard_id,
                    hazard_name=config["description"],
                    severity=config["severity"],
                    icon=config["icon"],
                    confidence=confidence,
                    matched_keywords=matches_a + matches_b,
                )
        
        return None

    def _generate_explanation(self, hazard: HazardWarning, text: str) -> str:
        """Generate a detailed explanation for the identified hazard."""
        template = self.EXPLANATIONS.get(hazard.hazard_type, "")
        
        if hazard.hazard_type == "electrocution":
            # Find specific electrical and water elements mentioned
            electrical_words = [kw for kw in ["cord", "cable", "wire", "electrical", "power"] if kw in text]
            water_words = [kw for kw in ["water", "puddle", "wet", "liquid", "moisture"] if kw in text]
            damage_words = [kw for kw in ["frayed", "damaged", "exposed", "broken", "worn"] if kw in text]
            
            electrical_str = ", ".join(electrical_words) if electrical_words else "electrical equipment"
            water_str = ", ".join(water_words) if water_words else "water/liquid"
            
            if damage_words:
                damage_detail = f"The {', '.join(damage_words)} condition of the wiring increases the risk of electrical arcing and short circuits"
            else:
                damage_detail = "Even intact electrical equipment near water poses a serious electrocution risk"
            
            return template.format(
                electrical_elements=electrical_str,
                water_elements=water_str,
                damage_detail=damage_detail,
            )
        elif hazard.hazard_type == "electrical_fire":
            damage_words = [kw for kw in ["frayed", "damaged", "exposed", "broken", "worn"] if kw in text]
            damage_str = ", ".join(damage_words) if damage_words else "damage"
            return template.format(damage_elements=damage_str)
        elif hazard.hazard_type == "dynamic_hazard":
            return (
                f"**{hazard.icon} {hazard.hazard_name} (Identified dynamically)**\n\n"
                f"The image analysis has dynamically identified a potential safety hazard: **{hazard.hazard_name}**.\n\n"
                f"**Contextual details:**\n"
                f"- The scene shows: *{', '.join(hazard.matched_keywords)}*\n"
                f"- The severity is assessed as **{hazard.severity}** based on the visual evidence.\n\n"
                f"Please review the situation carefully and take appropriate safety precautions."
            )
        
        return template

    def _check_dynamic_hazard(self, caption: str, vqa_answers: dict) -> Optional[HazardWarning]:
        """Dynamically detect and construct a hazard based on VQA answers."""
        # Check if the VQA answers indicate a hazard is present
        unsafe_ans1 = vqa_answers.get("Is there any hazard, danger, or unsafe condition in this image?", "").lower()
        unsafe_ans2 = vqa_answers.get("Is this scene dangerous?", "").lower()
        
        is_unsafe = False
        if any(w in unsafe_ans1 for w in ["yes", "unsafe", "danger", "hazard", "warning"]) or \
           any(w in unsafe_ans2 for w in ["yes", "dangerous", "hazard", "unsafe"]):
            is_unsafe = True
            
        if not is_unsafe:
            return None
            
        # Get the specific danger object
        danger_obj = vqa_answers.get("What is the specific dangerous object or situation?", "").strip()
        if not danger_obj or danger_obj.lower() in ["no", "none", "nothing", "n/a", "not sure", "unknown"]:
            danger_obj = caption
            
        if not danger_obj or danger_obj.lower() in ["no", "none", "nothing"]:
            danger_obj = "unspecified safety risk"
            
        danger_name = danger_obj[0].upper() + danger_obj[1:] if danger_obj else "Safety Risk"
        if len(danger_name) > 60:
            danger_name = danger_name[:57] + "..."
            
        # Get severity
        severity_ans = vqa_answers.get("What is the severity of this danger: low, medium, high, or critical?", "").lower()
        severity = "MEDIUM"
        for level in ["critical", "high", "medium", "low"]:
            if level in severity_ans:
                severity = level.upper()
                break
                
        # Get recommended action
        rec_action = vqa_answers.get("What is the immediate recommended action?", "").strip()
        if rec_action.lower() in ["no", "none", "nothing", "n/a", "not sure", "unknown"]:
            rec_action = ""
        if rec_action:
            rec_action = rec_action[0].upper() + rec_action[1:]
            
        # Map icon based on keywords
        icon = "⚠️"
        danger_lower = danger_name.lower()
        if any(w in danger_lower for w in ["fire", "smoke", "burn", "heat", "flame"]):
            icon = "🔥"
        elif any(w in danger_lower for w in ["electric", "wire", "voltage", "current", "plug", "cord"]):
            icon = "⚡"
        elif any(w in danger_lower for w in ["water", "slip", "wet", "puddle", "liquid", "spill"]):
            icon = "🚶"
        elif any(w in danger_lower for w in ["knife", "sharp", "blade", "cut", "scissors"]):
            icon = "🔪"
        elif any(w in danger_lower for w in ["fall", "height", "ladder", "climb", "roof"]):
            icon = "🪜"
        elif any(w in danger_lower for w in ["chemical", "toxic", "poison", "acid", "gas"]):
            icon = "🧪"
            
        return HazardWarning(
            hazard_type="dynamic_hazard",
            hazard_name=f"Dynamic: {danger_name}" if "hazard" not in danger_lower and "risk" not in danger_lower else danger_name,
            severity=severity,
            icon=icon,
            confidence=0.85,
            matched_keywords=[danger_obj],
            dynamic_action=rec_action,
        )
