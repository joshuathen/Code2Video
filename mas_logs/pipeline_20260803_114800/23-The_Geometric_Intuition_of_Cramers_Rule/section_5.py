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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Symmetry for y and Higher Dimensions", [
            "To find y, replace vector B with vector W.",
            "The same area ratio logic applies to find y.",
            "In 3D, we compare volumes of parallelepipeds instead."
        ])
        
        # Define colors
        COLOR_A = "#00FF00" # Green
        COLOR_B = "#0000FF" # Blue
        COLOR_W = "#FF0000" # Red
        COLOR_FORMULA = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Coordinate System setup
        plane = NumberPlane(
            x_range=[0, 4, 1], y_range=[0, 3, 1],
            x_length=3.5, y_length=2.5,
            background_line_style={"stroke_opacity": 0.4}
        )
        
        v_a = Arrow(plane.c2p(0,0,0), plane.c2p(1.5, 0.2, 0), buff=0, color=COLOR_A)
        v_b = Arrow(plane.c2p(0,0,0), plane.c2p(0.5, 1.2, 0), buff=0, color=COLOR_B)
        v_w = Arrow(plane.c2p(0,0,0), plane.c2p(2.0, 1.4, 0), buff=0, color=COLOR_W)
        
        label_a = MathTex(r"\vec{A}", color=COLOR_A, font_size=24).next_to(v_a.get_end(), RIGHT, buff=0.1)
        label_b = MathTex(r"\vec{B}", color=COLOR_B, font_size=24).next_to(v_b.get_end(), UP, buff=0.1)
        label_w = MathTex(r"\vec{W}", color=COLOR_W, font_size=24).next_to(v_w.get_end(), UP, buff=0.1)

        poly_aw = Polygon(
            plane.c2p(0,0,0), plane.c2p(1.5, 0.2, 0), plane.c2p(3.5, 1.6, 0), plane.c2p(2.0, 1.4, 0),
            stroke_width=0, fill_opacity=0.3, fill_color=COLOR_W
        )

        coord_sys = VGroup(plane, v_a, v_b, v_w, label_a, label_b, label_w, poly_aw)
        self.place_in_area(coord_sys, "C3", "E6")

        self.lecture[0].set_color(YELLOW)
        self.play(Create(plane), GrowArrow(v_a), GrowArrow(v_b), Write(label_a), Write(label_b))
        self.wait(1)
        
        # Replace B with W
        self.play(
            ReplacementTransform(v_b, v_w),
            ReplacementTransform(label_b, label_w)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Formula
        formula_y = MathTex(r"y = \frac{\det(\vec{A}, \vec{W})}{\det(\vec{A}, \vec{B})}", color=COLOR_FORMULA, font_size=32)
        # Issue 30 Fix: Updated position to B5 for better alignment
        self.place_at_grid(formula_y, "B5")
        
        self.play(FadeIn(poly_aw), Write(formula_y))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Clear previous elements
        self.play(FadeOut(coord_sys), FadeOut(formula_y))
        
        # 3D representation (Mock)
        origin = np.array([0, 0, 0])
        u = np.array([1.2, 0.2, 0])
        v = np.array([0.4, 0.8, 0])
        w = np.array([-0.3, 0.5, 0]) # simulated depth
        
        p0, p1, p2, p3 = origin, u, u + v, v
        p4, p5, p6, p7 = w, u + w, u + v + w, v + w
        
        back_faces = VGroup(
            Polygon(p4, p5, p6, p7, color=BLUE, fill_opacity=0.1),
            Line(p0, p4, color=BLUE_E, stroke_width=1),
            Line(p1, p5, color=BLUE_E, stroke_width=1),
            Line(p2, p6, color=BLUE_E, stroke_width=1),
            Line(p3, p7, color=BLUE_E, stroke_width=1),
        )
        front_face = Polygon(p0, p1, p2, p3, color=BLUE, fill_opacity=0.2)
        edges = VGroup(
            Line(p0, p1), Line(p1, p2), Line(p2, p3), Line(p3, p0),
            Line(p4, p5), Line(p5, p6), Line(p6, p7), Line(p7, p4)
        ).set_color(BLUE_A)
        
        volume_obj = VGroup(back_faces, front_face, edges)
        # Issue 28 Fix: Moved volume_obj to area C3-E6 to avoid overlap with lecture text
        self.place_in_area(volume_obj, "C3", "E6", scale_factor=1.5)
        
        vol_label = Text("Volume Ratio (3D)", font_size=24, color=WHITE)
        # Issue 29 Fix: Centered label at F5
        self.place_at_grid(vol_label, "F5")
        
        self.play(Create(volume_obj), Write(vol_label))
        self.wait(2)
        
        self.lecture[2].set_color(WHITE)
        self.wait(2)
