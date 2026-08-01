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
        # Define content
        title = "Prerequisite 2: The Inverse Pythagorean Theorem"
        lines = [
            "In a right triangle, altitude h relates to legs.",
            "The identity: one over h-squared equals reciprocal square sum.",
            "This lets us combine two light sources into one."
        ]
        
        # Colors
        COLOR_H = "#00FFFF"  # Cyan
        COLOR_A = "#FF00FF"  # Magenta
        COLOR_B = "#FFFF00"  # Yellow
        HIGHLIGHT = WHITE
        
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT))
        
        # Define triangle vertices
        C_pos = self.grid['D2']
        A_pos = self.grid['B2'] # Vertex along leg b
        B_pos = self.grid['D5'] # Vertex along leg a
        
        # Lines for the triangle
        leg_b = Line(C_pos, A_pos, color=COLOR_B)
        leg_a = Line(C_pos, B_pos, color=COLOR_A)
        hypotenuse = Line(A_pos, B_pos, color=WHITE)
        
        # Calculate altitude point D on hypotenuse
        AB_vec = B_pos - A_pos
        AC_vec = C_pos - A_pos
        t = np.dot(AC_vec, AB_vec) / np.dot(AB_vec, AB_vec)
        D_pos = A_pos + t * AB_vec
        
        altitude_h = Line(C_pos, D_pos, color=COLOR_H)
        
        # Right angle indicators
        ra_c = RightAngle(leg_b, leg_a, length=0.2, quadrant=(1,-1), color=WHITE)
        ra_d = RightAngle(Line(D_pos, A_pos), altitude_h, length=0.2, quadrant=(1,1), color=WHITE)

        # Labels - Using Text to avoid LaTeX dependency errors
        label_a = Text("a", color=COLOR_A, font="serif", slant=ITALIC)
        # Fix: Issue 35 - Reposition label 'a' to E4 for better centering
        self.place_at_grid(label_a, "E4", scale_factor=0.6)
        label_a.shift(DOWN * 0.2)
        
        label_b = Text("b", color=COLOR_B, font="serif", slant=ITALIC)
        # Fix: Issue 34 - Scale label 'b' to 0.5 to avoid crowding
        self.place_at_grid(label_b, "C1", scale_factor=0.5)
        label_b.shift(LEFT * 0.2)
        
        label_h = Text("h", color=COLOR_H, font="serif", slant=ITALIC)
        h_mid = (C_pos + D_pos) / 2
        label_h.move_to(h_mid + RIGHT * 0.3 + UP * 0.1).scale(0.6)

        # Animation Group 1
        self.play(Create(leg_a), Create(leg_b), Create(hypotenuse), run_time=1.5)
        self.play(Create(ra_c), Write(label_a), Write(label_b))
        self.play(Create(altitude_h), Create(ra_d), Write(label_h))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(HIGHLIGHT)
        )
        
        # Equation: 1/h² = 1/a² + 1/b²
        equation = VGroup(
            Text("1/h²", font="serif", font_size=36),
            Text("=", font="serif", font_size=36),
            Text("1/a²", font="serif", font_size=36),
            Text("+", font="serif", font_size=36),
            Text("1/b²", font="serif", font_size=36)
        ).arrange(RIGHT, buff=0.15)
        
        # Color specific variables
        equation[0][2].set_color(COLOR_H) # 'h'
        equation[2][2].set_color(COLOR_A) # 'a'
        equation[4][2].set_color(COLOR_B) # 'b'
        
        # Fix: Issue 33 - Move equation to A4-B6 and rescale to 0.9
        self.place_in_area(equation, "A4", "B6", scale_factor=0.9)

        # Fix: Issue 27 - Incorporate light asset icon
        light_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/light.svg", color=WHITE)
        self.place_at_grid(light_icon, "A3", scale_factor=0.5)
        
        self.play(Write(equation), FadeIn(light_icon))
        self.play(Indicate(equation, color=HIGHLIGHT))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(HIGHLIGHT)
        )
        
        # Visual cue for "combining light sources"
        glow_a = leg_a.copy().set_stroke(width=10, opacity=0.5)
        glow_b = leg_b.copy().set_stroke(width=10, opacity=0.5)
        glow_h = altitude_h.copy().set_stroke(width=10, opacity=0.5)
        
        self.play(
            FadeIn(glow_a), FadeIn(glow_b),
            equation[2:].animate.set_color(WHITE), 
            run_time=1
        )
        self.play(
            ReplacementTransform(VGroup(glow_a, glow_b), glow_h),
            equation[0].animate.set_color(WHITE),
            run_time=1.5
        )
        self.play(FadeOut(glow_h), FadeOut(light_icon))
        
        self.wait(3)
