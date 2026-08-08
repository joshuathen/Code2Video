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
        self.setup_layout("Closure: The Safety Net", [
            "Closure is our mathematical safety net.",
            "Results must remain inside the set.",
            "Exiting the set breaks the space."
        ])
        
        # Initialize lecture line colors to GRAY except the first one
        for i in range(1, len(self.lecture)):
            self.lecture[i].set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # A white rectangle represents a fenced-in playground.
        # Spans columns 3 to 5 (x from 2.0 to 5.0) and rows B to E (y from 1.7 to -2.3)
        playground = Rectangle(width=3.0, height=4.0, color=WHITE)
        self.place_in_area(playground, "B3", "E5")
        
        playground_label = Text("The Set (Vector Space V)", font_size=18, color=WHITE)
        self.place_at_grid(playground_label, "A4")
        
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Create(playground), Write(playground_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Green dot stays inside the fence during addition.
        dot_u = Dot(color=GREEN)
        dot_v = Dot(color=GREEN)
        self.place_at_grid(dot_u, "D4")
        self.place_at_grid(dot_v, "C5")
        
        label_u = MathTex(r"\vec{u}", font_size=24, color=GREEN).next_to(dot_u, DOWN, buff=0.1)
        label_v = MathTex(r"\vec{v}", font_size=24, color=GREEN).next_to(dot_v, DOWN, buff=0.1)
        
        self.play(self.lecture[0].animate.set_color(GRAY), self.lecture[1].animate.set_color(GREEN))
        self.play(FadeIn(dot_u, label_u), FadeIn(dot_v, label_v))
        self.wait(0.5)
        
        # Show addition result inside the playground at C4
        dot_w = Dot(color=GREEN)
        self.place_at_grid(dot_w, "C4")
        label_w = MathTex(r"\vec{u} + \vec{v} \in V", font_size=24, color=GREEN).next_to(dot_w, UP, buff=0.1)
        
        self.play(
            dot_u.animate.move_to(self.grid["C4"]),
            dot_v.animate.move_to(self.grid["C4"]),
            FadeOut(label_u, label_v),
            FadeIn(dot_w, label_w)
        )
        self.remove(dot_u, dot_v)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Dot moves outside fence and turns red via scalar.
        # Moving to column 1 which is well outside the rectangle (which starts at column 3)
        dot_out = Dot(color=RED)
        self.place_at_grid(dot_out, "C1")
        label_out = MathTex(r"c \cdot \vec{u} \notin V", font_size=24, color=RED).next_to(dot_out, DOWN, buff=0.1)
        
        self.play(self.lecture[1].animate.set_color(GRAY), self.lecture[2].animate.set_color(RED))
        
        self.play(
            dot_w.animate.move_to(self.grid["C1"]).set_color(RED),
            ReplacementTransform(label_w, label_out)
        )
        self.remove(dot_w)
        self.add(dot_out)
        
        # Add visual cross to indicate broken property
        failure_cross = Cross(dot_out, stroke_color=RED, scale_factor=0.5)
        self.play(Create(failure_cross))
        self.wait(2)
