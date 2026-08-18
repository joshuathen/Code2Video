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
        self.setup_layout("The Objective: What are we measuring?", [
            "Neural networks begin as random guesses.",
            "We use a cost function to measure error.",
            "High cost means the guess is very wrong."
        ])
        
        # Setup Archery components
        target = Circle(radius=0.5, color=YELLOW).set_stroke(width=3)
        self.place_at_grid(target, "C4", scale_factor=0.9)
        bullseye = Dot(color=RED)
        target.add(bullseye)
        
        # Terminal anchor icon (B007)
        anchor = Star(color=GREEN, fill_opacity=1, stroke_width=0).scale(0.2)
        anchor.move_to(target.get_center())
        
        # Arrow (the guess)
        arrow = Arrow(start=UP*2, end=ORIGIN, color=BLUE, buff=0)
        vector_group = VGroup(arrow, target, anchor)
        
        self.add(target, anchor)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Create(arrow))
        # Jittery random movement
        for _ in range(5):
            self.play(arrow.animate.shift(np.random.uniform(-0.5, 0.5, 3)), run_time=0.3)
        self.wait(1) 

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        distance_line = Line(arrow.get_end(), target.get_center(), color=YELLOW, stroke_width=4)
        cost_label = Text("Cost", font_size=20, color=YELLOW)
        self.place_at_grid(cost_label, "C3", scale_factor=0.7)
        
        self.play(Create(distance_line), Write(cost_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED)
        
        # Update vector positions as requested
        self.play(
            self.place_in_area(vector_group, "B3", "D4", scale_factor=0.8).animate,
            run_time=1
        )
        
        # Show High cost
        self.play(
            arrow.animate.shift(RIGHT*1.5 + DOWN*1),
            run_time=1
        )
        self.play(
            distance_line.animate.put_start_and_end_on(arrow.get_end(), target.get_center()),
            run_time=0.5
        )
        self.wait(1)
