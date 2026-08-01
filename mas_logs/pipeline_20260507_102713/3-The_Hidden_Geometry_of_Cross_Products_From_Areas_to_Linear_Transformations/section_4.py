from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup layout
        lecture_lines = [
            'The cross product defines a special linear transformation.',
            "Dotting it with u gives the parallelepiped's volume.",
            'This scalar triple product equals the 3x3 determinant.',
            'The cross product is the unique dual to this volume.',
            'It encodes how the transformation scales 3D space.'
        ]
        self.setup_layout("The Linear Transformation Perspective", lecture_lines)
        
        # Colors
        COLOR_U = "#C792EA"
        COLOR_V = "#58C4DD"
        COLOR_W = "#83C167"
        COLOR_VXW = "#F8B195"
        COLOR_TEXT = "#FFFF00"
        
        # Center of geometry area B2-D4
        geom_center = (self.grid["B2"] + self.grid["D4"]) / 2
        origin = geom_center + np.array([-0.6, -0.6, 0])
        
        vec_v_coord = np.array([1.2, 0.2, 0])
        vec_w_coord = np.array([0.4, 0.6, 0])
        vec_u_coord = np.array([-0.2, 1.2, 0])
        
        v_pt = origin + vec_v_coord
        w_pt = origin + vec_w_coord
        u_pt = origin + vec_u_coord
        vw_pt = v_pt + vec_w_coord
        vu_pt = v_pt + vec_u_coord
        wu_pt = w_pt + vec_u_coord
        vwu_pt = v_pt + w_pt + vec_u_coord

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_V))
        
        v_vec = Arrow(origin, v_pt, buff=0, color=COLOR_V)
        w_vec = Arrow(origin, w_pt, buff=0, color=COLOR_W)
        v_label = Text("v", color=COLOR_V, font_size=20).next_to(v_pt, DOWN, buff=0.1)
        w_label = Text("w", color=COLOR_W, font_size=20).next_to(w_pt, LEFT, buff=0.1)
        
        base_parallelogram = Polygon(origin, v_pt, vw_pt, w_pt, color=COLOR_V, stroke_width=2, fill_opacity=0.1)
        
        self.play(Create(v_vec), Create(w_vec), Write(v_label), Write(w_label))
        self.play(Create(base_parallelogram))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_U)
        )
        
        u_vec = Arrow(origin, u_pt, buff=0, color=COLOR_U)
        u_label = Text("u", color=COLOR_U, font_size=20).next_to(u_pt, UP, buff=0.1)
        
        # Parallelepiped faces
        faces = VGroup(
            Polygon(u_pt, vu_pt, vwu_pt, wu_pt, color=COLOR_U, stroke_width=1, fill_opacity=0.15), # top
            Polygon(origin, v_pt, vu_pt, u_pt, color=COLOR_U, stroke_width=1, fill_opacity=0.1), # front
            Polygon(w_pt, vw_pt, vwu_pt, wu_pt, color=COLOR_U, stroke_width=1, fill_opacity=0.1), # back
            Polygon(origin, w_pt, wu_pt, u_pt, color=COLOR_U, stroke_width=1, fill_opacity=0.1), # left
            Polygon(v_pt, vw_pt, vwu_pt, vu_pt, color=COLOR_U, stroke_width=1, fill_opacity=0.1)  # right
        )
        
        vol_text = Text("Volume = det(u, v, w)", color=COLOR_TEXT, font_size=20)
        # Fix for Issue 32: Reposition vol_text to A4 to avoid overlap
        self.place_at_grid(vol_text, 'A4', scale_factor=0.8)

        self.play(Create(u_vec), Write(u_label))
        self.play(FadeIn(faces), Write(vol_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_TEXT)
        )
        
        triple_product = Text("(v x w) · u = det(u, v, w)", color=WHITE, font_size=24)
        # Fix for Issue 33: Reposition triple_product formula to avoid clutter
        self.place_in_area(triple_product, 'E2', 'E6', scale_factor=0.7)
        
        self.play(Write(triple_product))
        self.play(triple_product.animate.set_color(COLOR_TEXT), run_time=0.5)
        self.play(triple_product.animate.set_color(WHITE), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_VXW)
        )
        
        # Cross product vector (visualized as perpendicular to v and w)
        vxw_coord = np.array([0.1, 1.4, 0])
        vxw_pt = origin + vxw_coord
        vxw_vec = Arrow(origin, vxw_pt, buff=0, color=COLOR_VXW)
        vxw_label = Text("v x w", color=COLOR_VXW, font_size=20).next_to(vxw_pt, RIGHT, buff=0.1)
        
        self.play(Create(vxw_vec), Write(vxw_label))
        
        # Dot product visualization (dashed line from u to vxw)
        dot_line = DashedLine(u_pt, vxw_pt, color=WHITE, stroke_width=2)
        self.play(Create(dot_line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_TEXT)
        )
        
        dual_note = Text("v x w is the dual of the determinant", color=COLOR_TEXT, font_size=18)
        # Fix for Issue 34: Improve horizontal balance of dual_note
        self.place_in_area(dual_note, 'F2', 'F6', scale_factor=0.6)
        
        self.play(Write(dual_note))
        self.play(
            triple_product.animate.scale(1.2).set_color(COLOR_TEXT),
            vol_text.animate.set_color(WHITE),
            run_time=1
        )
        self.wait(2)
