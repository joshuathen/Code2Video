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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup the layout with the specific lines for section 3
        title = "The Core Concept: The Borsuk-Ulam Theorem"
        lines = [
            "Antipodal points are exactly opposite each other on spheres.",
            "The Borsuk-Ulam theorem states their function values match.",
            "Two opposite Earth locations share temperature and pressure."
        ]
        self.setup_layout(title, lines)

        # Color definitions for visual alignment
        COL_L1 = BLUE_B
        COL_L2 = WHITE
        COL_L3 = GREEN_B
        COL_DOT = "#FF0000"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COL_L1))
        
        # Load and place Earth asset (Asset integration Issue 26)
        # Issue 31: Place in B3-E6 area to avoid lecture text
        earth = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/earth.svg")
        self.place_in_area(earth, 'B3', 'E6', scale_factor=1.2)
        earth_center = earth.get_center()
        
        # Antipodal points x and -x (Issue 26)
        radius = 1.2
        p1_coord = earth_center + np.array([radius * np.cos(PI/4), radius * np.sin(PI/4), 0])
        p2_coord = earth_center + np.array([radius * np.cos(PI/4 + PI), radius * np.sin(PI/4 + PI), 0])
        
        dot1 = Dot(p1_coord, color=COL_DOT)
        dot2 = Dot(p2_coord, color=COL_DOT)
        
        label_x = Text("x", color=COL_L1, font_size=24).next_to(dot1, UR, buff=0.1)
        label_mx = Text("-x", color=COL_L1, font_size=24).next_to(dot2, DL, buff=0.1)
        
        self.play(FadeIn(earth))
        self.play(FadeIn(dot1), FadeIn(dot2), Write(label_x), Write(label_mx))
        
        # Show a subtle line connecting them to highlight they are antipodal
        connecting_line = Line(p1_coord, p2_coord, color=COL_L1, stroke_opacity=0.2)
        self.play(Create(connecting_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COL_L2))
        
        # Issue 33: Place formula in A3-A5 with scale 1.0
        formula = Text("f(x) = f(-x)", color=COL_L2)
        self.place_in_area(formula, 'A3', 'A5', scale_factor=1.0)
        
        # Issue 32: Place mapping notation at B2 with scale 0.8
        f_label = Text("f: Sn -> Rn", color=COL_L2, font_size=20)
        self.place_at_grid(f_label, 'B2', scale_factor=0.8)
        
        self.play(Write(formula), Write(f_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COL_L3))
        
        # Earth example data labels matching the description
        earth_data1 = VGroup(
            Text("Temp: 20C", font_size=16, color=COL_L3),
            Text("Press: 1013hPa", font_size=16, color=COL_L3)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(dot1, RIGHT, buff=0.2)
        
        earth_data2 = VGroup(
            Text("Temp: 20C", font_size=16, color=COL_L3),
            Text("Press: 1013hPa", font_size=16, color=COL_L3)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(dot2, LEFT, buff=0.2)
        
        self.play(
            FadeIn(earth_data1),
            FadeIn(earth_data2)
        )
        self.wait(3)
