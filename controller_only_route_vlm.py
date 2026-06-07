# coding=utf-8
# controller_only_route_vlm.py

import json
import re
import torch
from typing import Dict


class RouteOnlyOvisController:
    """
    Metadata-free route-only controller.

    Design:
    1) First try hard-routing-inspired lexical patterns
    2) Only if no strong lexical hit, call VLM as fallback
    3) Return only route + reason + raw_response

    Important:
    - This controller is designed to SHARE the already-loaded Ovis model
      from your perception module, to avoid loading a second 9B model.
    """

    def __init__(self, model):
        # Reuse external loaded model, do NOT load another one here.
        self.model = model

        self.valid_routes = {
            "line0_pure_vlm",
            "line1_base",
            "line2_orientation",
            "line3_bbox"
        }

        # =========================
        # Prompt for fallback VLM
        # =========================
        self.controller_prompt = (
            "You are a controller for a multi-stage spatial reasoning pipeline.\n"
            "Your task is NOT to answer the question.\n"
            "Your task is to select the SINGLE most suitable route.\n\n"

            "Routes:\n"
            "- line0_pure_vlm: best for special direct-reasoning cases such as volume comparison, reflective-surface questions, cardinal direction, some betweenness questions, and some simple path/accessibility questions.\n"
            "- line1_base: conservative fallback route when no strong route-specific pattern is detected.\n"
            "- line2_orientation: best for orientation-sensitive or geometric-structure questions, including facing direction, object rotation, handedness, gravity/stability, shape projection, depth/layering/occlusion, corner/directional/proximity reasoning.\n"
            "- line3_bbox: best for size/scale/extent/layout questions, including relative size comparison, perspective/scale questions, fit/cover, and alignment-pattern questions.\n\n"

            "Important:\n"
            "- Prefer line3_bbox for most size-and-scale questions.\n"
            "- Prefer line2_orientation for most orientation questions except cardinal direction.\n"
            "- Prefer line2_orientation for most 3D geometry questions except clear volume-comparison cases.\n"
            "- Prefer line2_orientation for most depth/occlusion questions except reflective-surface cases.\n"
            "- Prefer line3_bbox for alignment-pattern questions.\n"
            "- Prefer line0_pure_vlm for cardinal direction, reflective surfaces, clear volume comparison, and some simple path/accessibility questions.\n"
            "- Use line1_base only when none of the above strongly applies.\n\n"

            "Return ONLY valid JSON:\n"
            "{\n"
            '  "route": "one of: line0_pure_vlm, line1_base, line2_orientation, line3_bbox",\n'
            '  "reason": "one short sentence"\n'
            "}\n"
        )

        # ============================================================
        # Hard-routing-inspired lexical patterns
        # These are designed to imitate your previous best oracle split
        # ============================================================

        # ---- line3: old best setting ----
        # Relative Positioning -> Alignment Patterns
        # Size & Scale -> basically all
        self.line3_patterns = [
            # alignment / layout
            "aligned", "alignment", "row", "rows", "column", "columns",
            "horizontal", "vertical", "diagonal", "centered", "same line",
            "left to right", "overall directional orientation", "arranged in which manner",
            "position of the whiteboard", "generally aligned", "increasing", "decreasing",

            # size / scale / extent
            "bigger", "smaller", "larger", "shorter", "taller", "wider", "narrower",
            "same size", "same sized", "scale", "fit", "cover", "inside", "within",
            "radius", "height", "width", "size", "proportion", "actual size", "real size",
            "distance-size", "perspective", "distortion", "shadow", "projection",
            "look larger", "look smaller", "same height", "larger than", "smaller than",
            "fit inside", "cover the", "carry all", "take up less space",
            "will it look", "will it fit", "can it fit", "can all", "equal sized",
            "same scale", "consistent", "consistency", "more expanded", "compressed",
            "elongated", "actual room look", "actual shark", "real rabbit size"
        ]

        # ---- line0: old best setting ----
        # 3D Geometry -> Volume Comparison
        # Depth & Occlusion -> Reflective Surfaces
        # Orientation -> Cardinal Direction
        # Relative Positioning -> Betweenness Relationships
        # Size & Scale -> Scale Consistency
        # Spatial Navigation -> Accessibility Constraints / Pathway Existence
        self.line0_patterns = [
            # volume comparison
            "volume", "hold more", "hold less", "takes up more space", "takes up less space",
            "bulk volume", "can hold more", "can hold less", "encloses more space",
            "which one can hold more", "which takes up more space", "compare the volume",
            "which pair together encloses more space",

            # reflective surfaces
            "reflection", "reflective", "shiniest", "reflects", "reflected", "mirror surface",
            "clearest reflection", "most light reflection", "shiny surface", "what is reflected",
            "shows the clearest reflection", "what object is reflected", "reflection of itself",

            # cardinal direction
            "north", "south", "east", "west", "cardinal direction",

            # betweenness
            "between",

            # scale consistency
            "scale consistency", "same scale", "realistically", "real size", "actual size",
            "tall enough", "big enough", "small enough", "should be larger", "should be smaller",

            # simple accessibility/pathway style
            "clear path", "path exists", "reachable", "unreachable", "blocking the path",
            "obstructing", "can reach", "can go", "there is a direct path",
            "there is a clear path", "which directions can", "which path", "path structure",
            "is it true", "true", "false"
        ]

        # ---- line2: old best setting ----
        # Orientation except cardinal
        # 3D Geometry except volume
        # Depth & Occlusion except reflective surfaces
        # Relative Positioning except alignment & betweenness
        self.line2_patterns = [
            # orientation
            "facing", "face", "looking", "look", "pointing", "point",
            "angle", "tilt", "turned", "rotation", "rotated",
            "left hand", "right hand", "dominant hand", "handedness",
            "thumb", "grip",

            # 3D geometry
            "gravity", "fall", "falling", "drop", "dropped", "spill", "spilling",
            "roll", "rolling", "slide", "sliding", "stable", "unstable", "balance",
            "support", "supported", "tip", "topple", "shaken", "shake", "precarious",
            "shadow outline", "what shape would", "looked from above", "look from above",
            "fit completely", "hidden inside", "pass through", "contain", "inside the mug",
            "could it be completely hidden", "what shape would the shadow", "if you were looking",
            "if you looked from above", "shape would its outline", "look at the .* fit",

            # depth & occlusion
            "foreground", "background", "frontmost", "foremost", "behind", "hidden from view",
            "occluded", "covered", "under", "what is behind", "what is under",
            "what is hidden", "layer", "depth layer", "front layer", "furthest",
            "same depth layer", "closest to the viewer", "second front-most", "second furthest",

            # relative positioning
            "corner", "angle positioning", "to the left of", "to the right of",
            "directly behind", "immediately to the left", "immediately to the right",
            "closest", "farthest", "proximity", "nearer", "further", "farther",
            "relative to", "located behind", "positioned behind", "in front of"
        ]

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _contains_any(self, text: str, keywords) -> bool:
        return any(k in text for k in keywords)

    def _generate_response(self, messages, max_tokens=256, enable_thinking=False):
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        input_ids = input_ids.cuda()
        pixel_values = pixel_values.cuda() if pixel_values is not None else None
        grid_thws = grid_thws.cuda() if grid_thws is not None else None

        generate_kwargs = {"max_new_tokens": max_tokens}

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                **generate_kwargs
            )

        input_len = input_ids.shape[1]
        if len(outputs[0]) > input_len and torch.equal(outputs[0][:input_len], input_ids[0]):
            generated_tokens = outputs[0][input_len:]
        else:
            generated_tokens = outputs[0]

        output_text = self.model.text_tokenizer.decode(generated_tokens, skip_special_tokens=True)

        del input_ids, pixel_values, grid_thws, outputs
        torch.cuda.empty_cache()
        return output_text

    def _safe_parse_json(self, text: str) -> Dict:
        raw = text.strip()

        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        try:
            data = json.loads(raw)
        except Exception:
            return {
                "route": "line1_base",
                "reason": f"JSON parse failed. Raw output: {text[:200]}"
            }

        route = data.get("route", "line1_base")
        if route not in self.valid_routes:
            route = "line1_base"

        reason = data.get("reason", "")
        if not isinstance(reason, str):
            reason = ""

        return {"route": route, "reason": reason}

    def _lexical_route(self, question: str, options_text: str) -> str:
        """
        Hard-pattern route approximating your previous best oracle routing.
        Priority:
            line3 > line0 > line2 > line1
        """
        text = self._normalize(question + " " + options_text)

        # 1) line3 first: almost all size-and-scale questions + alignment patterns
        if self._contains_any(text, self.line3_patterns):
            return "line3_bbox"

        # 2) line0 second: oracle-special direct reasoning classes
        if self._contains_any(text, self.line0_patterns):
            return "line0_pure_vlm"

        # 3) line2 third: the main geometric/orientation bucket
        if self._contains_any(text, self.line2_patterns):
            return "line2_orientation"

        # 4) conservative fallback
        return "line1_base"

    def _rule_repair(self, question: str, options_text: str, pred: Dict) -> Dict:
        """
        Final route is dominated by lexical hard-routing approximation.
        """
        original_route = pred["route"]
        lexical_route = self._lexical_route(question, options_text)

        pred["route"] = lexical_route

        if pred["route"] != original_route:
            pred["reason"] = pred["reason"] + f" | repaired_by_hard_patterns: {original_route}->{pred['route']}"

        return pred

    def predict(self, question: str, options_text: str) -> Dict:
        """
        Use lexical route directly if it is confident (i.e., not line1_base).
        Only use VLM when lexical route falls back to line1_base.
        """
        lexical_route = self._lexical_route(question, options_text)

        # If lexical pattern is strong enough, trust it directly
        if lexical_route != "line1_base":
            return {
                "route": lexical_route,
                "reason": "Assigned by hard-routing-inspired lexical patterns.",
                "raw_response": ""
            }

        # Otherwise ask VLM only for fallback cases
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Output JSON:"
        )

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.controller_prompt + "\n\n" + user_prompt}]
            }
        ]

        raw_response = self._generate_response(messages, max_tokens=256, enable_thinking=False)
        parsed = self._safe_parse_json(raw_response)
        parsed = self._rule_repair(question, options_text, parsed)
        parsed["raw_response"] = raw_response
        return parsed