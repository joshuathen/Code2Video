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
        # Setup layout
        title = "The Evolution of a Vector"
        lines = [
            "Vectors are more than simple arrows.",
            "They can evolve into digital signals.",
            "Specific rules define these abstract objects.",
            "Evolution highlights the power of structure.",
            "Meet Vector-Bot, our logical guide."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Show a white 2D arrow pointing NE on grid.
        self.lecture[0].set_color(WHITE)
        arrow = Arrow(ORIGIN, RIGHT + UP, buff=0, color=WHITE)
        self.place_in_area(arrow, "B2", "D4", scale_factor=1.5)
        
        self.play(Create(arrow))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Arrow morphs into a blue sine wave digital signal.
        self.lecture[1].set_color(BLUE_C)
        
        # Create sine wave
        sine_wave = FunctionGraph(
            lambda x: 0.5 * np.sin(2 * PI * x),
            x_range=[-1.5, 1.5],
            color=BLUE_C
        )
        self.place_in_area(sine_wave, "B2", "D4", scale_factor=1.0)
        
        self.play(
            self.lecture[0].animate.set_color(WHITE), 
            Transform(arrow, sine_wave),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display 'VECTORS = OBJECTS + RULES' at bottom center.
        self.lecture[2].set_color(WHITE)
        
        formula = MathTex(
            r"\text{VECTORS}", "=", r"\text{OBJECTS}", "+", r"\text{RULES}",
            color=WHITE
        )
        self.place_in_area(formula, "F1", "F6", scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight 'OBJECTS' in yellow and 'RULES' in teal (equivalent to cyan).
        self.lecture[3].set_color(YELLOW)
        
        self.play(
            formula[2].animate.set_color(YELLOW),
            formula[4].animate.set_color(TEAL),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # A simple robot face appears next to the wave.
        self.lecture[4].set_color(WHITE)
        
        # Construct a simple robot face (Vector-Bot)
        head = Square(side_length=0.8, color=WHITE)
        eye_l = Dot(radius=0.05, color=WHITE).shift(LEFT*0.2 + UP*0.1)
        eye_r = Dot(radius=0.05, color=WHITE).shift(RIGHT*0.2 + UP*0.1)
        mouth = Line(LEFT*0.2 + DOWN*0.2, RIGHT*0.2 + DOWN*0.2, color=WHITE)
        robot_bot = VGroup(head, eye_l, eye_r, mouth)
        
        # Position Vector-Bot next to the wave
        self.place_at_grid(robot_bot, "C5", scale_factor=0.8)
        
        self.play(FadeIn(robot_bot, shift=UP))
        self.wait(2)
