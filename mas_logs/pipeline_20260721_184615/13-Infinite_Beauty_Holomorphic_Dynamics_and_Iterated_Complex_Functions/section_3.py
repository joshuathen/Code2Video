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
        title = "The Core Equation: f(z) = z² + c"
        lines = [
            "The dynamic begins with the formula z squared plus c.",
            "Here, c acts as a constant shift or nudge.",
            "Squaring rotates and stretches our current position.",
            "Adding c pushes the point in a new direction.",
            "This repeated process generates a complex path or orbit."
        ]
        
        self.setup_layout(title, lines)
        
        # Hex Colors as per L008
        WHITE_HEX = "#FFFFFF"
        BLUE_HEX = "#0000FF"
        RED_HEX = "#FF0000"
        GREEN_HEX = "#00FF00"
        
        # === Animation for Lecture Line 1 ===
        # Fade in the equation 'f(z) = z^2 + c' centered on screen in white (#FFFFFF).
        self.lecture[0].set_color(WHITE_HEX)
        # Using segments for highlighting later: f( z ) = z ^2 + c
        # Indices: 0:f(, 1:z, 2:), 3:=, 4:z, 5:^2, 6:+, 7:c
        equation = MathTex("f(", "z", ")", "=", "z", "^2", "+", "c", color=WHITE_HEX)
        # [Issue 26 Fix]
        self.place_in_area(equation, 'A2', 'B5', scale_factor=1.3)
        
        self.play(FadeIn(equation))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # "Here, c acts as a constant shift or nudge."
        # Place a red SurroundingRectangle (#FF0000) around the constant 'c'
        self.lecture[1].set_color(RED_HEX)
        nudge_rect = SurroundingRectangle(equation[7], color=RED_HEX, buff=0.1)
        
        self.play(Create(nudge_rect))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # "Squaring rotates and stretches our current position."
        # Place a blue SurroundingRectangle (#0000FF) around the variable 'z'
        self.lecture[2].set_color(BLUE_HEX)
        # Highlight the z part that is being squared (indices 4 and 5: z^2)
        pos_rect = SurroundingRectangle(equation[4:6], color=BLUE_HEX, buff=0.1)
        
        self.play(
            ReplacementTransform(nudge_rect.copy(), pos_rect),
            nudge_rect.animate.set_stroke(opacity=0.3)
        )
        self.add(pos_rect)
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # "Adding c pushes the point in a new direction."
        self.lecture[3].set_color(RED_HEX)
        # Highlight the nudge (+ c) part (indices 6 and 7)
        plus_c_rect = SurroundingRectangle(equation[6:8], color=RED_HEX, buff=0.1)
        
        self.play(
            FadeOut(pos_rect),
            ReplacementTransform(nudge_rect, plus_c_rect)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # "This repeated process generates a complex path or orbit."
        self.lecture[4].set_color(GREEN_HEX)
        
        # Sequentially display the text 'z0 -> z1 -> z2' with white arrows (#FFFFFF)
        orbit_text = MathTex("z_0", "\\to", "z_1", "\\to", "z_2", color=WHITE_HEX)
        # [Issue 27 Fix]
        self.place_in_area(orbit_text, 'C2', 'C5', scale_factor=0.8)
        
        self.play(Write(orbit_text))
        
        # L010 and Storyboard: Wait 2.0s before visualization
        self.wait(2.0) 
        
        # Coordinate system for jumps (E2-F6 area)
        # L001: Configure tips in constructor
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=3,
            axis_config={"include_tip": True, "color": WHITE_HEX}
        )
        # [Issue 28 Fix]
        self.place_in_area(axes, 'D2', 'F5', scale_factor=0.9)
        
        dot = Dot(color=GREEN_HEX)
        dot.move_to(axes.c2p(0, 0))
        
        # Illustrative orbit jumps: (0,0) -> (1,1) -> (0,2) -> (-1.5, 0.5)
        path_points = [
            axes.c2p(0, 0),
            axes.c2p(1, 1),
            axes.c2p(0, 2),
            axes.c2p(-1.5, 0.5)
        ]
        
        self.play(Create(axes), FadeIn(dot))
        
        for i in range(len(path_points) - 1):
            start = path_points[i]
            end = path_points[i+1]
            line = Line(start, end, color=GREEN_HEX, stroke_width=2)
            self.play(
                dot.animate.move_to(end),
                Create(line),
                run_time=1
            )
            
        self.wait(2.0)
