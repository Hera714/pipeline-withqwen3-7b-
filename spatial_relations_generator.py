# coding=utf-8
# spatial_relations_generator.py

def _get_egocentric_direction(obj_a, obj_b):
    # [这部分辅助函数代码保持不变，直接复用你原有的即可]
    pos_a = obj_a.get('global_3d_position', {})
    pos_b = obj_b.get('global_3d_position', {})
    
    if not pos_a or not pos_b:
        return None
        
    dx = pos_b.get('X_meters', 0) - pos_a.get('X_meters', 0)
    dz = pos_b.get('Z_meters', 0) - pos_a.get('Z_meters', 0)
    
    facing = obj_a.get('2d_pointing_direction', '').lower()
    left_right = ""
    front_back = ""

    if 'camera' in facing:
        left_right = "LEFT" if dx > 0 else "RIGHT"
        front_back = "IN FRONT OF" if dz < 0 else "BEHIND"
    elif 'away' in facing:
        left_right = "RIGHT" if dx > 0 else "LEFT"
        front_back = "IN FRONT OF" if dz > 0 else "BEHIND"
    elif 'left' in facing:
        left_right = "LEFT" if dz > 0 else "RIGHT"
        front_back = "IN FRONT OF" if dx < 0 else "BEHIND"
    elif 'right' in facing:
        left_right = "LEFT" if dz < 0 else "RIGHT"
        front_back = "IN FRONT OF" if dx > 0 else "BEHIND"
    else:
        return None 

    return f"{left_right} and {front_back}"


def generate_spatial_relations(scene_graph, target_perspective):
    if not scene_graph:
        return "No objects detected."
        
    relations = []
    
    # ==============================================================
    # 1. 独立物体姿态推断 (Single Object Geometric States)
    # ==============================================================
    relations.append("=== Single Object Geometric States ===")
    state_found = False
    for obj in scene_graph:
        name = obj['name']
        profile = obj.get('visible_profile', '').lower()
        direction = obj.get('2d_pointing_direction', '').lower()
        
        if 'diagonal' in profile:
            relations.append(f"- [{name}]: Presents a '{profile}' view pointing '{direction}'. This indicates the object is ROTATED/ANGLED (oblique to the camera's line-of-sight).")
            state_found = True
        elif profile == 'side_profile':
            relations.append(f"- [{name}]: Presents a strict '{profile}' view pointing '{direction}'. This indicates it is positioned laterally (side-ways).")
            state_found = True
        elif profile in ['frontal', 'back']:
            relations.append(f"- [{name}]: Presents a '{profile}' view. It is aligned with the camera axis.")
            state_found = True
            
    if not state_found:
        relations.append("- No specific geometric profile (frontal/side/diagonal) could be clearly determined for the objects.")

    # ==============================================================
    # 2. 多物体交互关系
    # ==============================================================
    if len(scene_graph) >= 2:
        relations.append("\n=== Absolute Spatial Relationships (Camera Perspective) ===")
        abs_rel_found = False
        for i in range(len(scene_graph)):
            for j in range(i + 1, len(scene_graph)):
                a, b = scene_graph[i], scene_graph[j]
                pos_a, pos_b = a.get('global_3d_position', {}), b.get('global_3d_position', {})
                if not pos_a or not pos_b: continue
                
                # Depth
                if pos_a['Z_meters'] < pos_b['Z_meters']:
                    relations.append(f"- [Depth]: {a['name']} is CLOSER to the camera than {b['name']}.")
                    abs_rel_found = True
                elif pos_a['Z_meters'] > pos_b['Z_meters']:
                    relations.append(f"- [Depth]: {b['name']} is CLOSER to the camera than {a['name']}.")
                    abs_rel_found = True
                # Horizontal
                if pos_a['X_meters'] < pos_b['X_meters']:
                    relations.append(f"- [Horizontal]: {a['name']} is to the LEFT of {b['name']}.")
                    abs_rel_found = True
                elif pos_a['X_meters'] > pos_b['X_meters']:
                    relations.append(f"- [Horizontal]: {b['name']} is to the LEFT of {a['name']}.")
                    abs_rel_found = True
                    
        if not abs_rel_found:
            relations.append("- No significant depth or horizontal coordinate differences detected.")

        # ====== 核心逻辑：Pairwise 几何与朝向推导 ======
        # (这段你原本写的如果 deductions 为空就不打标题，逻辑非常安全，直接保留即可)
        deductions = []
        for i in range(len(scene_graph)):
            for j in range(i + 1, len(scene_graph)):
                # ... [保留原有的 deductions 生成逻辑，代码过长省略] ...
                a, b = scene_graph[i], scene_graph[j]
                p_a, p_b = a.get('visible_profile', ''), b.get('visible_profile', '')
                d_a, d_b = a.get('2d_pointing_direction', ''), b.get('2d_pointing_direction', '')
                
                is_a_ortho = p_a in ['frontal', 'back']
                is_b_ortho = p_b in ['frontal', 'back']
                is_a_side = p_a == 'side_profile'
                is_b_side = p_b == 'side_profile'
                is_a_diag = 'diagonal' in p_a
                is_b_diag = 'diagonal' in p_b

                # --- 组 1: 几何形态判定 (Perpendicular, Parallel, Oblique) ---
                if (is_a_ortho and is_b_side) or (is_b_ortho and is_a_side):
                    deductions.append(f"- [Perpendicularity]: '{a['name']}' is '{p_a}' and '{b['name']}' is '{p_b}'. This strict geometric discrepancy dictates they are PERPENDICULAR (at ~90-degree angle) to each other.")
                elif is_a_diag and is_b_diag and d_a in ['left', 'right'] and d_b in ['left', 'right'] and d_a != d_b:
                    deductions.append(f"- [Perpendicularity/Intersection]: Both '{a['name']}' and '{b['name']}' present a 'diagonal' view but point in diverging directions ('{d_a}' vs '{d_b}'). This dictates they are mounted PERPENDICULAR to each other or intersect at a significant angle.")
                elif is_a_side and is_b_side:
                    deductions.append(f"- [Parallelism]: Both '{a['name']}' and '{b['name']}' present a strict 'side_profile'. Their longitudinal axes are PARALLEL.")
                elif is_a_ortho and is_b_ortho:
                    deductions.append(f"- [Parallelism]: Both '{a['name']}' and '{b['name']}' present a '{p_a}'/'{p_b}' view. Their front planes are PARALLEL.")
                elif is_a_diag and is_b_diag and d_a == d_b and d_a not in ['', 'unknown']:
                    deductions.append(f"- [Similar Orientation]: Both '{a['name']}' and '{b['name']}' are 'diagonal' and point '{d_a}'. Their orientations are SIMILAR or GENERALLY ALIGNED (though not necessarily strictly parallel).")
                elif (is_a_diag and (is_b_ortho or is_b_side)) or (is_b_diag and (is_a_ortho or is_a_side)):
                    deductions.append(f"- [Alignment/Rotation]: One is '{p_a}' and the other is '{p_b}'. They are NOT PARALLEL; the diagonal object is ROTATED/ANGLED forming an oblique angle relative to the other.")

                # --- 组 2: 矢量朝向判定 (Same, Opposite, Facing) ---
                valid_dirs = ['left', 'right', 'camera', 'away', 'up', 'down']
                if d_a in valid_dirs and d_b in valid_dirs:
                    if d_a == d_b:
                        deductions.append(f"- [Relative Direction]: Both point '{d_a}'. They are facing the SAME direction.")
                    else:
                        if (d_a == 'left' and d_b == 'right') or (d_a == 'right' and d_b == 'left'):
                            x_a = a.get('global_3d_position', {}).get('X_meters', 0)
                            x_b = b.get('global_3d_position', {}).get('X_meters', 0)
                            if (d_a == 'right' and x_a < x_b) or (d_a == 'left' and x_a > x_b):
                                deductions.append(f"- [Relative Direction]: '{a['name']}' points '{d_a}' and '{b['name']}' points '{d_b}', and they are physically oriented toward each other. They are FACING EACH OTHER.")
                            else:
                                deductions.append(f"- [Relative Direction]: '{a['name']}' points '{d_a}' while '{b['name']}' points '{d_b}'. They are pointing in OPPOSITE / DIVERGING directions.")
                        elif (d_a == 'camera' and d_b == 'away') or (d_a == 'away' and d_b == 'camera') or (d_a == 'up' and d_b == 'down') or (d_a == 'down' and d_b == 'up'):
                            deductions.append(f"- [Relative Direction]: '{a['name']}' points '{d_a}' while '{b['name']}' points '{d_b}'. They are pointing in OPPOSITE / DIVERGING directions.")

        if deductions:
            relations.append("\n=== Pairwise Geometric & Orientation Deductions ===")
            relations.extend(deductions)

    # ==============================================================
    # 3. 动态他我视角映射 (Egocentric)
    # ==============================================================
    if target_perspective and target_perspective.lower() != "camera" and len(scene_graph) >= 2:
        relations.append(f"\n=== Egocentric Perspective (From the view of '{target_perspective}') ===")
        center_obj = next((obj for obj in scene_graph if obj['name'].lower() == target_perspective.lower()), None)
                
        if center_obj and center_obj.get('2d_pointing_direction') not in ['unknown', '']:
            for obj_b in scene_graph:
                if obj_b == center_obj: continue
                ego_dir = _get_egocentric_direction(center_obj, obj_b)
                if ego_dir:
                    relations.append(f"- From {center_obj['name']}'s own perspective, the {obj_b['name']} is to its {ego_dir}.")
        else:
            relations.append(f"- Could not establish the exact facing direction for '{target_perspective}'. Cannot deduce egocentric relations.")

    return "\n".join(relations)