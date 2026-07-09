import numpy as np
import torch
import xatlas
import trimesh
import moderngl
from PIL import Image
from PIL import ImageFilter
from pathlib import Path
import cv2


def make_atlas(mesh, texture_resolution, texture_padding):
    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices, mesh.faces)
    options = xatlas.PackOptions()
    options.resolution = texture_resolution
    options.padding = texture_padding
    options.bilinear = True
    atlas.generate(pack_options=options)
    vmapping, indices, uvs = atlas[0]
    return {
        "vmapping": vmapping,
        "indices": indices,
        "uvs": uvs,
    }


def rasterize_position_and_normal_atlas(
    mesh, atlas_vmapping, atlas_indices, atlas_uvs, texture_resolution, texture_padding
):
    ctx = moderngl.create_context(standalone=True)
    basic_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_uv;
            in vec3 in_pos;
            in vec3 in_normal;
            out vec3 v_pos;
            out vec3 v_normal;
            void main() {
                v_pos = in_pos;
                v_normal = in_normal;
                gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 v_pos;
            in vec3 v_normal;
            layout(location=0) out vec4 o_pos;
            layout(location=1) out vec4 o_normal;
            void main() {
                o_pos = vec4(v_pos, 1.0);
                o_normal = vec4(v_normal, 1.0);
            }
        """,
    )
    gs_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_uv;
            in vec3 in_pos;
            in vec3 in_normal;
            out vec3 vg_pos;
            out vec3 vg_normal;
            void main() {
                vg_pos = in_pos;
                vg_normal = in_normal;
                gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0);
            }
        """,
        geometry_shader="""
            #version 330
            uniform float u_resolution;
            uniform float u_dilation;
            layout (triangles) in;
            layout (triangle_strip, max_vertices = 12) out;
            in vec3 vg_pos[];
            in vec3 vg_normal[];
            out vec3 vf_pos;
            out vec3 vf_normal;
            void lineSegment(int aidx, int bidx) {
                vec2 a = gl_in[aidx].gl_Position.xy;
                vec2 b = gl_in[bidx].gl_Position.xy;
                vec3 aPos = vg_pos[aidx];
                vec3 bPos = vg_pos[bidx];
                vec3 aNorm = vg_normal[aidx];
                vec3 bNorm = vg_normal[bidx];

                vec2 dir = normalize((b - a) * u_resolution);
                vec2 offset = vec2(-dir.y, dir.x) * u_dilation / u_resolution;

                gl_Position = vec4(a + offset, 0.0, 1.0);
                vf_pos = aPos;
                vf_normal = aNorm;
                EmitVertex();
                gl_Position = vec4(a - offset, 0.0, 1.0);
                vf_pos = aPos;
                vf_normal = aNorm;
                EmitVertex();
                gl_Position = vec4(b + offset, 0.0, 1.0);
                vf_pos = bPos;
                vf_normal = bNorm;
                EmitVertex();
                gl_Position = vec4(b - offset, 0.0, 1.0);
                vf_pos = bPos;
                vf_normal = bNorm;
                EmitVertex();
            }
            void main() {
                lineSegment(0, 1);
                lineSegment(1, 2);
                lineSegment(2, 0);
                EndPrimitive();
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 vf_pos;
            in vec3 vf_normal;
            layout(location=0) out vec4 o_pos;
            layout(location=1) out vec4 o_normal;
            void main() {
                o_pos = vec4(vf_pos, 1.0);
                o_normal = vec4(vf_normal, 1.0);
            }
        """,
    )
    uvs = atlas_uvs.flatten().astype("f4")
    pos = mesh.vertices[atlas_vmapping].flatten().astype("f4")
    
    # Calculate normals if not present
    if not hasattr(mesh, 'vertex_normals') or len(mesh.vertex_normals) == 0:
        mesh_copy = mesh.copy()
        try:
            mesh_copy.fix_normals()
        except:
            pass
        normals = mesh_copy.vertex_normals[atlas_vmapping].flatten().astype("f4")
    else:
        normals = mesh.vertex_normals[atlas_vmapping].flatten().astype("f4")
        
    indices = atlas_indices.flatten().astype("i4")
    
    vbo_uvs = ctx.buffer(uvs)
    vbo_pos = ctx.buffer(pos)
    vbo_normal = ctx.buffer(normals)
    ibo = ctx.buffer(indices)
    
    vao_content = [
        vbo_uvs.bind("in_uv", layout="2f"),
        vbo_pos.bind("in_pos", layout="3f"),
        vbo_normal.bind("in_normal", layout="3f"),
    ]
    
    basic_vao = ctx.vertex_array(basic_prog, vao_content, ibo)
    gs_vao = ctx.vertex_array(gs_prog, vao_content, ibo)
    
    fbo = ctx.framebuffer(
        color_attachments=[
            ctx.texture((texture_resolution, texture_resolution), 4, dtype="f4"),
            ctx.texture((texture_resolution, texture_resolution), 4, dtype="f4")
        ]
    )
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    
    gs_prog["u_resolution"].value = texture_resolution
    gs_prog["u_dilation"].value = texture_padding
    gs_vao.render()
    basic_vao.render()

    fbo_pos_bytes = fbo.color_attachments[0].read()
    fbo_pos_np = np.frombuffer(fbo_pos_bytes, dtype="f4").reshape(
        texture_resolution, texture_resolution, 4
    )
    
    fbo_normal_bytes = fbo.color_attachments[1].read()
    fbo_normal_np = np.frombuffer(fbo_normal_bytes, dtype="f4").reshape(
        texture_resolution, texture_resolution, 4
    )
    
    return fbo_pos_np, fbo_normal_np


def positions_to_colors(model, scene_code, positions_texture, texture_resolution):
    device = scene_code.device
    positions = torch.tensor(positions_texture.reshape(-1, 4)[:, :-1]).to(device)
    with torch.no_grad():
        queried_grid = model.renderer.query_triplane(
            model.decoder,
            positions,
            scene_code,
        )
    rgb_f = queried_grid["color"].cpu().numpy().reshape(-1, 3)
    rgba_f = np.insert(rgb_f, 3, positions_texture.reshape(-1, 4)[:, -1], axis=1)
    rgba_f[rgba_f[:, -1] == 0.0] = [0, 0, 0, 0]
    return rgba_f.reshape(texture_resolution, texture_resolution, 4)


def get_projected_colors(positions_texture, normals_texture, image_pil, cam_dist: float = 1.9, fovy_deg: float = 40.0):
    """
    Project world positions back to the input image and sample colors exactly.
    Camera parameters `cam_dist` and `fovy_deg` can be tuned to match TripoSR's render.
    Returns: projected_colors (H,W,3), facing_weight (H,W), valid (H,W boolean)
    """
    image_rgba = image_pil.convert("RGBA")
    img_w, img_h = image_rgba.size
    img_np = np.array(image_rgba).astype(np.float32) / 255.0
    img_rgb = img_np[:, :, :3]
    img_alpha = img_np[:, :, 3] > 0.5

    pos = positions_texture[:, :, :3]
    mask = positions_texture[:, :, 3] > 0

    # Camera intrinsics from provided fovy
    fovy_rad = float(fovy_deg) * np.pi / 180.0
    tan_half_fov = np.tan(fovy_rad / 2.0)
    fx = 0.5 * img_w / tan_half_fov
    fy = 0.5 * img_h / tan_half_fov

    # Transform world positions into the camera coordinate system.
    # Camera is at (cam_dist, 0, 0), looking toward the origin.
    x = pos[:, :, 0]
    y = pos[:, :, 1]
    z = pos[:, :, 2]
    x_cam = y
    y_cam = z
    z_cam = x - cam_dist

    # Only keep points in front of the camera.
    in_front = z_cam < 0

    # Project to image space using pinhole model
    u = img_w * 0.5 + fx * (x_cam / -z_cam)
    v = img_h * 0.5 + fy * (y_cam / -z_cam)

    # Only accept points inside the image and in front of the camera.
    valid_uv = (u >= 0) & (u <= img_w - 1) & (v >= 0) & (v <= img_h - 1) & in_front

    # Use nearest-neighbor sampling so selected image pixel colors are preserved exactly.
    u_nn = np.clip(np.rint(u).astype(np.int32), 0, img_w - 1)
    v_nn = np.clip(np.rint(v).astype(np.int32), 0, img_h - 1)
    proj_colors = img_rgb[v_nn, u_nn]

    image_mask = img_alpha[v_nn, u_nn]
    valid = mask & valid_uv & image_mask

    projected_colors = np.zeros_like(pos)
    projected_colors[valid] = proj_colors[valid]

    # facing weight as diagnostic (higher means more front-facing)
    facing_weight = np.clip((z_cam + cam_dist) / cam_dist, 0.0, 1.0)
    return projected_colors, facing_weight, valid, u, v


def _project_with_mapping(positions_texture, image_pil, cam_dist, fovy_deg, idx_map, sign_map):
    """Project using an explicit axis mapping and sign flips.
    idx_map: tuple of indices into pos (0=x,1=y,2=z) specifying which pos component maps to camera x,y,z
    sign_map: tuple of 1 or -1 to flip signs for each mapped component
    Returns: proj_colors, facing_weight, valid, u, v
    """
    image_rgba = image_pil.convert("RGBA")
    img_w, img_h = image_rgba.size
    img_np = np.array(image_rgba).astype(np.float32) / 255.0
    img_rgb = img_np[:, :, :3]
    img_alpha = img_np[:, :, 3] > 0.5

    pos = positions_texture[:, :, :3]
    mask = positions_texture[:, :, 3] > 0

    fovy_rad = float(fovy_deg) * np.pi / 180.0
    tan_half_fov = np.tan(fovy_rad / 2.0)
    fx = 0.5 * img_w / tan_half_fov
    fy = 0.5 * img_h / tan_half_fov

    # map components
    a = pos[:, :, idx_map[0]] * sign_map[0]
    b = pos[:, :, idx_map[1]] * sign_map[1]
    c = pos[:, :, idx_map[2]] * sign_map[2]

    # interpret mapped components as camera x_cam=a, y_cam=b, z_cam=c - cam_dist
    x_cam = a
    y_cam = b
    z_cam = c - cam_dist
    in_front = z_cam < 0

    u = img_w * 0.5 + fx * (x_cam / -z_cam)
    v = img_h * 0.5 + fy * (y_cam / -z_cam)

    valid_uv = (u >= 0) & (u <= img_w - 1) & (v >= 0) & (v <= img_h - 1) & in_front

    # Use nearest-neighbor sampling so selected image pixel colors are preserved exactly.
    u_nn = np.clip(np.rint(u).astype(np.int32), 0, img_w - 1)
    v_nn = np.clip(np.rint(v).astype(np.int32), 0, img_h - 1)

    proj_colors = img_rgb[v_nn, u_nn]

    image_mask = img_alpha[v_nn, u_nn]
    valid = mask & valid_uv & image_mask

    projected_colors = np.zeros_like(pos)
    projected_colors[valid] = proj_colors[valid]

    facing_weight = np.clip((z_cam + cam_dist) / cam_dist, 0.0, 1.0)
    return projected_colors, facing_weight, valid, u, v


def find_best_axis_mapping(positions_texture, image_pil, cam_dist=1.9, fovy_deg=40.0):
    """Try permutations and sign flips to find mapping that gives best projection match.
    Score by valid count and color MSE on valid pixels.
    Returns best proj_colors, facing_weight, valid, chosen_map, chosen_signs
    """
    from itertools import permutations, product

    best_score = (-1e9, None)
    best_result = None
    perms = list(permutations((0, 1, 2)))
    signs = list(product((1, -1), repeat=3))
    image_rgba = image_pil.convert("RGBA")
    img_np = np.array(image_rgba).astype(np.float32) / 255.0
    img_rgb = img_np[:, :, :3]

    for p in perms:
        for s in signs:
            try:
                pc, fw, vmask, u, v = _project_with_mapping(positions_texture, image_pil, cam_dist, fovy_deg, p, s)
                cnt = int(np.sum(vmask))
                if cnt == 0:
                    continue
                # compute mse between projected colors and image at corresponding u,v
                u0 = np.floor(u[vmask]).astype(np.int32)
                v0 = np.floor(v[vmask]).astype(np.int32)
                proj_vals = pc[vmask]
                img_vals = img_rgb[v0, u0]
                mse = float(np.mean((proj_vals - img_vals) ** 2))
                # score prefer larger cnt and lower mse
                score = (cnt) - 1000.0 * mse
                if score > best_score[0]:
                    best_score = (score, (p, s, cnt, mse))
                    best_result = (pc, fw, vmask, p, s, cnt, mse)
            except Exception:
                continue

    if best_result is None:
        return None
    pc, fw, vmask, p, s, cnt, mse = best_result
    print(f"[BAKE] axis-mapping chosen {p} signs {s} cnt={cnt} mse={mse:.6f}")
    return pc, fw, vmask, p, s


def _find_best_axis_mapping_region_params(positions_texture, image_pil, region_mask_image, cam_dist, fovy_deg):
    from itertools import permutations, product

    image_rgba = image_pil.convert("RGBA")
    img_w, img_h = image_rgba.size
    img_np = np.array(image_rgba).astype(np.float32) / 255.0
    img_rgb = img_np[:, :, :3]

    best_score = -1e9
    best_result = None
    perms = list(permutations((0, 1, 2)))
    signs = list(product((1, -1), repeat=3))

    for p in perms:
        for s in signs:
            try:
                pc, fw, vmask, u, v = _project_with_mapping(positions_texture, image_pil, cam_dist, fovy_deg, p, s)
                if u is None:
                    continue
                u0 = np.floor(u).astype(np.int32).clip(0, img_w - 1)
                v0 = np.floor(v).astype(np.int32).clip(0, img_h - 1)
                sampled = region_mask_image[v0, u0]
                sel = vmask & sampled
                cnt = int(np.sum(sel))
                if cnt == 0:
                    continue
                proj_vals = pc[sel]
                img_vals = img_rgb[v0[sel], u0[sel]]
                mse = float(np.mean((proj_vals - img_vals) ** 2))
                score = cnt - 1000.0 * mse
                if score > best_score:
                    best_score = score
                    best_result = (pc, fw, vmask, u, v, p, s, cnt, mse)
            except Exception:
                continue

    return best_result


def find_best_axis_mapping_region(
    positions_texture,
    image_pil,
    region_mask_image,
    cam_dists=None,
    fovy_deg=None,
):
    """Search for the best axis mapping and camera intrinsics for the given region mask.
    region_mask_image: boolean numpy array shaped (H_img, W_img) True where region is present.
    """
    if cam_dists is None:
        cam_dists = [1.0, 1.3, 1.6, 1.9, 2.2, 2.6]
    elif isinstance(cam_dists, (int, float)):
        cam_dists = [cam_dists]
    if fovy_deg is None:
        fovy_deg = [30.0, 35.0, 40.0, 45.0, 50.0]
    elif isinstance(fovy_deg, (int, float)):
        fovy_deg = [fovy_deg]

    best_score = -1e9
    best_result = None
    best_cam = None
    best_fovy = None

    for cd in cam_dists:
        for fv in fovy_deg:
            result = _find_best_axis_mapping_region_params(
                positions_texture, image_pil, region_mask_image, cd, fv
            )
            if result is None:
                continue
            pc, fw, vmask, u, v, p, s, cnt, mse = result
            score = cnt - 1000.0 * mse
            if score > best_score:
                best_score = score
                best_result = (pc, fw, vmask, u, v, p, s, cnt, mse)
                best_cam = cd
                best_fovy = fv

    if best_result is None:
        return None

    pc, fw, vmask, u, v, p, s, cnt, mse = best_result
    print(
        f"[BAKE] axis-mapping(region) chosen {p} signs {s} cam_dist={best_cam} fovy={best_fovy} cnt={cnt} mse={mse:.6f}"
    )
    return pc, fw, vmask, u, v, p, s


def bake_texture(mesh, model, scene_code, texture_resolution, original_image_path=None):
    texture_padding = round(max(2, texture_resolution / 256))
    atlas = make_atlas(mesh, texture_resolution, texture_padding)
    positions_texture, normals_texture = rasterize_position_and_normal_atlas(
        mesh,
        atlas["vmapping"],
        atlas["indices"],
        atlas["uvs"],
        texture_resolution,
        texture_padding,
    )
    
    # 1. Get base color from AI prediction (triplane)
    colors_texture = positions_to_colors(
        model, scene_code, positions_texture, texture_resolution
    )
    
    # 2. If original image is provided, project it and blend
    if original_image_path is not None:
        try:
            image_pil = Image.open(original_image_path).convert("RGBA")
            proj_colors, facing_weight, valid, u, v = get_projected_colors(
                positions_texture, normals_texture, image_pil
            )
            
            valid_count = int(np.sum(valid))
            total_count = int(valid.size)
            valid_ratio = valid_count / total_count if total_count > 0 else 0.0
            print(f"[BAKE] projected valid pixels: {valid_count}/{total_count} ({valid_ratio*100:.2f}%)")

            # If coverage is low, try a small grid search over camera distance and fovy
            if valid_ratio < 0.35:
                try:
                    best = (valid_count, None, None, None, None, None)
                    cand_cam = [1.0, 1.3, 1.6, 1.9, 2.2, 2.6]
                    cand_fov = [30.0, 35.0, 40.0, 45.0, 50.0]
                    for cd in cand_cam:
                        for fv in cand_fov:
                            pc, fw, vmask, uu, vv = get_projected_colors(positions_texture, normals_texture, image_pil, cam_dist=cd, fovy_deg=fv)
                            cnt = int(np.sum(vmask))
                            if cnt > best[0]:
                                best = (cnt, pc, fw, vmask, cd, fv)
                    if best[0] > valid_count:
                        print(f"[BAKE] Improved projection using cam_dist={best[4]}, fovy={best[5]} -> {best[0]}/{total_count} ({best[0]/total_count*100:.2f}%)")
                        proj_colors, facing_weight, valid = best[1], best[2], best[3]
                        valid_count = int(best[0])
                        valid_ratio = valid_count / total_count
                except Exception as e:
                    print(f"[BAKE] camera search failed: {e}")

            # If coverage remains low, try axis-permutation + sign flips to recover mapping
            if valid_ratio < 0.35:
                try:
                    mapping = find_best_axis_mapping(positions_texture, image_pil)
                    if mapping is not None:
                        pc2, fw2, vmask2, p, s = mapping
                        cnt2 = int(np.sum(vmask2))
                        if cnt2 > valid_count:
                            print(f"[BAKE] axis-mapping improved projection using map={p} signs={s} -> {cnt2}/{total_count} ({cnt2/total_count*100:.2f}%)")
                            proj_colors, facing_weight, valid = pc2, fw2, vmask2
                            valid_count = cnt2
                            valid_ratio = valid_count / total_count
                except Exception as e:
                    print(f"[BAKE] axis mapping search failed: {e}")

            # Save projection mask, heatmap and overlay for debugging
            try:
                out_dir = Path(original_image_path).parent
                stem = Path(original_image_path).stem
                mask_img = (valid.astype(np.uint8) * 255)
                Image.fromarray(mask_img).convert("L").save(str(out_dir / f"{stem}_projection_mask.png"))

                # heatmap from facing_weight (0-1) -> 0-255
                heat = (np.clip(facing_weight, 0.0, 1.0) * 255).astype(np.uint8)
                Image.fromarray(heat).convert("L").save(str(out_dir / f"{stem}_projection_heatmap.png"))

                # overlay original image with red tint where valid
                orig = Image.open(original_image_path).convert("RGBA")
                orig_arr = np.array(orig)
                overlay = orig_arr.copy()
                # ensure mask shape matches image
                mask_resized = mask_img.astype(bool)
                if mask_resized.shape != overlay.shape[:2]:
                    # try to resize mask to image size
                    from PIL import Image as PILImage
                    mask_resized = np.array(PILImage.fromarray(mask_img).resize((overlay.shape[1], overlay.shape[0]))) > 0
                overlay[mask_resized, :3] = np.clip(overlay[mask_resized, :3].astype(np.int32) + np.array([160, 0, 0]), 0, 255)
                Image.fromarray(overlay.astype(np.uint8)).save(str(out_dir / f"{stem}_projection_overlay.png"))
                # also save a blurred soft mask used for blending
                try:
                    soft = np.array(Image.fromarray(mask_img).filter(ImageFilter.GaussianBlur(radius=3))).astype(np.uint8)
                    Image.fromarray(soft).convert("L").save(str(out_dir / f"{stem}_projection_mask_blur.png"))
                except Exception:
                    pass
            except Exception as e:
                print(f"[BAKE] Could not save projection debug images: {e}")

            # Load region masks if available and apply per-region projection
            region_masks = {}
            for r in ("sole", "body", "collar"):
                pth = out_dir / f"{stem}_mask_{r}.png"
                if not pth.exists():
                    candidates = list(out_dir.glob(f"*{stem}*_mask_{r}.png"))
                    if len(candidates) == 0:
                        candidates = list(out_dir.glob(f"*mask_{r}.png"))
                    if candidates:
                        pth = candidates[0]
                        print(f"[BAKE] region mask fallback found for {r}: {pth.name}")
                if pth.exists():
                    try:
                        rm = np.array(Image.open(pth).convert("L")) > 0
                        region_masks[r] = rm
                    except Exception:
                        print(f"[BAKE] could not open region mask {pth}")
                        pass

            if region_masks:
                print(f"[BAKE] found region masks: {sorted(region_masks.keys())}")
            else:
                print("[BAKE] no region masks found for per-region projection")

            # compute integer sample coords for source image
            img_w, img_h = image_pil.size
            u0 = np.floor(u).astype(np.int32).clip(0, img_w - 1)
            v0 = np.floor(v).astype(np.int32).clip(0, img_h - 1)

            applied_any = False
            applied_mask = np.zeros_like(valid)
            # Try per-region axis-mapping first (higher precision)
            if len(region_masks) > 0:
                for r, rm in region_masks.items():
                    try:
                        # ensure mask matches image size
                        if rm.shape != (img_h, img_w):
                            from PIL import Image as PILImage
                            rm_resized = np.array(PILImage.fromarray((rm.astype(np.uint8) * 255)).resize((img_w, img_h))) > 0
                        else:
                            rm_resized = rm
                        mapping = find_best_axis_mapping_region(positions_texture, image_pil, rm_resized)
                        if mapping is not None:
                            pc2, fw2, vmask2, uu2, vv2, p, s = mapping
                            u0_r = np.floor(uu2).astype(np.int32).clip(0, img_w - 1)
                            v0_r = np.floor(vv2).astype(np.int32).clip(0, img_h - 1)
                            sampled = rm_resized[v0_r, u0_r]
                            region_valid = vmask2 & sampled
                            if np.any(region_valid):
                                colors_texture[region_valid, :3] = pc2[region_valid]
                                applied_mask = applied_mask | region_valid
                                applied_any = True
                                print(f"[BAKE] applied region mapping for {r}: cnt={int(np.sum(region_valid))}")
                    except Exception:
                        continue

            # fallback: apply global projection for any remaining valid pixels in region masks
            if len(region_masks) > 0:
                for r, rm in region_masks.items():
                    try:
                        if rm.shape != (img_h, img_w):
                            from PIL import Image as PILImage
                            rm_resized = np.array(PILImage.fromarray((rm.astype(np.uint8) * 255)).resize((img_w, img_h))) > 0
                        else:
                            rm_resized = rm
                        sampled = rm_resized[v0, u0]
                        region_valid = valid & sampled & ~applied_mask
                        if np.any(region_valid):
                            colors_texture[region_valid, :3] = proj_colors[region_valid]
                            applied_mask = applied_mask | region_valid
                            applied_any = True
                    except Exception:
                        continue

            # global fallback: soft-blend globally if nothing applied
            if not applied_any:
                try:
                    soft_mask = np.array(Image.fromarray(mask_img).filter(ImageFilter.GaussianBlur(radius=2))).astype(np.float32) / 255.0
                    soft_mask = soft_mask[:, :, None]
                    colors_texture[:, :, :3] = proj_colors * soft_mask + colors_texture[:, :, :3] * (1.0 - soft_mask)
                except Exception:
                    valid_mask = valid[:, :, np.newaxis]
                    colors_texture[:, :, :3] = np.where(valid_mask, proj_colors, colors_texture[:, :, :3])

            # If coverage below desired target, inpaint missing texels using OpenCV
            try:
                TARGET_COVERAGE = 0.90
                if valid_ratio < TARGET_COVERAGE:
                    print(f"[BAKE] coverage {valid_ratio*100:.2f}% < {TARGET_COVERAGE*100:.0f}% -> running inpaint")
                    # prepare uint8 BGR image and mask for inpaint
                    rgb_img = np.clip(colors_texture[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
                    bgr_img = rgb_img[:, :, ::-1].copy()
                    inpaint_mask = (~valid).astype(np.uint8) * 255
                    # inpaint expects single-channel mask of same HxW
                    inpainted = cv2.inpaint(bgr_img, inpaint_mask, 3, cv2.INPAINT_TELEA)
                    # convert back to RGB float32 0-1
                    inpainted_rgb = inpainted[:, :, ::-1].astype(np.float32) / 255.0
                    colors_texture[:, :, :3] = inpainted_rgb
                    # update valid to all true after inpaint
                    valid[:, :] = True
                    valid_count = int(np.sum(valid))
                    valid_ratio = valid_count / total_count
                    print(f"[BAKE] post-inpaint coverage: {valid_count}/{total_count} ({valid_ratio*100:.2f}%)")
                    # save debug inpaint image
                    try:
                        out_dir = Path(original_image_path).parent
                        stem = Path(original_image_path).stem
                        Image.fromarray((np.clip(colors_texture[:, :, :3], 0.0, 1.0) * 255).astype(np.uint8)).save(str(out_dir / f"{stem}_texture_inpaint.png"))
                    except Exception:
                        pass
            except Exception as e:
                print(f"[BAKE] inpaint failed: {e}")
            
        except Exception as e:
            print(f"[BAKE] Lỗi khi project texture từ ảnh gốc: {e}")
    
    return {
        "vmapping": atlas["vmapping"],
        "indices": atlas["indices"],
        "uvs": atlas["uvs"],
        "colors": colors_texture,
    }
