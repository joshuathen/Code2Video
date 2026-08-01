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
        # Section title and lecture lines
        title_text = "The Visual Proof: Linking Segments"
        lecture_lines = [
            "Pick any point P on the elliptical slice.",
            "Connect P to the foci F1 and F2.",
            "PF1 equals the distance to the upper contact circle.",
            "PF2 equals the distance to the lower contact circle.",
            "These segments align along the cone's slant height."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Assets paths
        ASSET_CONE = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg"
        ASSET_SPHERE = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"

        # Colors
        COLOR_P = "#FF0000"
        COLOR_BLUE = "#0000FF"
        COLOR_ORANGE = "#FF4500"
        COLOR_PLANE = "#FFFFE0"
        
        # Grid references
        g = self.grid

        # Geometrically consistent positions (Calculated to ensure PA=PF1 and PB=PF2)
        # and A, P, B are collinear on the slant height.
        p_pos = g["D5"]
        a_pos = g["B4"]
        b_pos = g["F6"]
        f1_pos = g["C3"]
        f2_pos = g["F4"]

        # === Animation for Lecture Line 1 ===
        # "Pick any point P on the elliptical slice."
        # Display the cone, plane, and spheres with a red point P (#FF0000) on the ellipse.
        
        # Cone SVG
        cone_svg = SVGMobject(ASSET_CONE, color=GREY_C, fill_opacity=0.2)
        self.place_in_area(cone_svg, "A3", "F6", scale_factor=2.5)
        
        # Spheres SVG (Addressing issues 36, 37, 42)
        sphere1 = SVGMobject(ASSET_SPHERE, color=WHITE, fill_opacity=0.3)
        self.place_in_area(sphere1, 'A4', 'B5', scale_factor=0.7)
        
        sphere2 = SVGMobject(ASSET_SPHERE, color=WHITE, fill_opacity=0.3)
        self.place_in_area(sphere2, 'D4', 'F5', scale_factor=0.8)
        
        # Plane line (as slice)
        plane_line = Line(g["C3"], g["F6"], color=COLOR_PLANE, stroke_width=2)
        
        # Point P
        dot_p = Dot(p_pos, color=COLOR_P)
        label_p = MathTex("P", color=COLOR_P, font_size=24).next_to(dot_p, RIGHT, buff=0.1)

        self.lecture[0].set_color(YELLOW)
        self.play(
            FadeIn(cone_svg),
            FadeIn(sphere1), FadeIn(sphere2),
            Create(plane_line),
            FadeIn(dot_p), Write(label_p),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Connect P to the foci F1 and F2."
        dot_f1 = Dot(f1_pos, color=WHITE)
        label_f1 = MathTex("F_1", color=WHITE, font_size=24).next_to(dot_f1, LEFT, buff=0.1)
        
        dot_f2 = Dot(f2_pos, color=WHITE)
        label_f2 = MathTex("F_2", color=WHITE, font_size=24).next_to(dot_f2, DOWN, buff=0.1)
        
        seg_pf1 = Line(p_pos, f1_pos, color=COLOR_BLUE)
        seg_pf2 = Line(p_pos, f2_pos, color=COLOR_ORANGE)

        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(
            FadeIn(dot_f1), Write(label_f1),
            FadeIn(dot_f2), Write(label_f2),
            Create(seg_pf1), Create(seg_pf2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "PF1 equals the distance to the upper contact circle."
        dot_a = Dot(a_pos, color=COLOR_BLUE)
        label_a = MathTex("A", color=COLOR_BLUE, font_size=24).next_to(dot_a, UP, buff=0.1)
        seg_pa = Line(p_pos, a_pos, color=COLOR_BLUE)

        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(
            FadeIn(dot_a), Write(label_a),
            Create(seg_pa)
        )
        # Highlight equality visually
        self.play(seg_pf1.animate.set_stroke(width=8), seg_pa.animate.set_stroke(width=8), run_time=0.5)
        self.play(seg_pf1.animate.set_stroke(width=4), seg_pa.animate.set_stroke(width=4), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "PF2 equals the distance to the lower contact circle."
        dot_b = Dot(b_pos, color=COLOR_ORANGE)
        label_b = MathTex("B", color=COLOR_ORANGE, font_size=24).next_to(dot_b, DOWN, buff=0.1)
        seg_pb = Line(p_pos, b_pos, color=COLOR_ORANGE)

        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(
            FadeIn(dot_b), Write(label_b),
            Create(seg_pb)
        )
        # Highlight equality visually
        self.play(seg_pf2.animate.set_stroke(width=8), seg_pb.animate.set_stroke(width=8), run_time=0.5)
        self.play(seg_pf2.animate.set_stroke(width=4), seg_pb.animate.set_stroke(width=4), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "These segments align along the cone's slant height."
        # Highlight the straight line AB through P
        seg_ab = Line(a_pos, b_pos, color=WHITE, stroke_width=2)
        
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.play(
            Create(seg_ab),
            seg_pa.animate.set_color(WHITE),
            seg_pb.animate.set_color(WHITE),
            run_time=2
        )
        self.play(Indicate(seg_ab, color=YELLOW), run_time=2)
        self.wait(2)
