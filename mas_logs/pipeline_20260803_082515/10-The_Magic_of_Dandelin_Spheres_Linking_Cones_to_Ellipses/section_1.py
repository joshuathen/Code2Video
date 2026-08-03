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

class Section1Scene(TeachingScene):
    def construct(self):
        # Fetching data from storyboard
        title = "The Mystery of the Sliced Cone"
        lecture_lines = [
            "Conic sections appear when a plane slices a cone.",
            "Tilting the plane transforms a circle into an ellipse.",
            "How can we prove this shape is truly an ellipse?"
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        GOLD = "#FFD700"
        LIGHT_BLUE = "#ADD8E6"
        ORANGE_RED = "#FF4500"
        MAGENTA = "#FF00FF"

        # === Animation for Lecture Line 1 ===
        # highlight lecture line 1 in GOLD
        self.play(self.lecture[0].animate.set_color(GOLD))
        
        # Cone (SVG Asset)
        # Issue 16: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg]
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg", color=GOLD, stroke_width=2)
        self.place_in_area(cone, "B3", "E5", scale_factor=2.0)
        
        # Horizontal plane
        plane = Rectangle(height=0.1, width=3.5, color=LIGHT_BLUE, fill_opacity=0.5, fill_color=LIGHT_BLUE)
        self.place_at_grid(plane, "C4")
        
        # Intersection (Circle as perspective ellipse)
        circle_intersection = Ellipse(width=1.6, height=0.4, color=WHITE).set_stroke(width=4)
        # Issue 27: self.place_at_grid(circle_intersection, 'C4', scale_factor=1.2)
        self.place_at_grid(circle_intersection, "C4", scale_factor=1.2)
        
        self.play(Create(cone))
        self.play(FadeIn(plane))
        self.play(Create(circle_intersection))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # highlight lecture line 2 in ORANGE_RED
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(ORANGE_RED)
        )
        
        # Tilted ellipse and foci group (foci will be revealed later)
        ellipse_at_origin = Ellipse(width=2.0, height=0.6, color=ORANGE_RED).set_stroke(width=4)
        ellipse_at_origin.rotate(20 * DEGREES)
        
        # Calculate offsets for foci
        f1_offset = np.array([-0.7 * np.cos(20*DEGREES), -0.7 * np.sin(20*DEGREES), 0])
        f2_offset = np.array([0.7 * np.cos(20*DEGREES), 0.7 * np.sin(20*DEGREES), 0])
        f1_dot = Dot(f1_offset, color=MAGENTA)
        f2_dot = Dot(f2_offset, color=MAGENTA)
        
        foci_group = VGroup(ellipse_at_origin, f1_dot, f2_dot)
        # Issue 26: self.place_in_area(foci_group, 'C3', 'C5', scale_factor=1.2)
        self.place_in_area(foci_group, 'C3', 'C5', scale_factor=1.2)
        
        # Transform circle into the ellipse part of the group
        self.play(
            plane.animate.rotate(20 * DEGREES),
            Transform(circle_intersection, ellipse_at_origin)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # highlight lecture line 3 in MAGENTA
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(MAGENTA)
        )
        
        foci_labels = VGroup(
            Text("F1", font_size=16, color=MAGENTA).next_to(f1_dot, DOWN, buff=0.1),
            Text("F2", font_size=16, color=MAGENTA).next_to(f2_dot, UP, buff=0.1)
        )
        
        formula = MathTex("PF_1 + PF_2 = \\text{Constant?}", font_size=24, color=MAGENTA)
        # Issue 25: self.place_in_area(formula, 'F3', 'F5')
        self.place_in_area(formula, 'F3', 'F5')
        
        self.play(FadeIn(f1_dot, f2_dot), Write(foci_labels))
        self.play(Write(formula))
        self.wait(2)
