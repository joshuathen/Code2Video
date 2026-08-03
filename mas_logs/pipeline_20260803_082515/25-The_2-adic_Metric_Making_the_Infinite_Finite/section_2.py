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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "In 2-adic math, \"closeness\" depends on divisibility by 2.",
            "Highly divisible numbers like 16 are considered \"small.\"",
            "Numbers like 3 are \"large\" because 2 doesn't divide them."
        ]
        self.setup_layout("Prerequisite: The Divisibility Rule (p-adic Valuation)", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Draw a funnel-like machine labeled '2-adic Sieve' #CCCCCC.
        funnel_outline = VMobject(color="#CCCCCC")
        funnel_outline.set_points_as_corners([
            [-0.7, 0.8, 0], [-0.2, -0.3, 0], [-0.2, -0.8, 0],
            [0.2, -0.8, 0], [0.2, -0.3, 0], [0.7, 0.8, 0]
        ])
        sieve_label = Text("2-adic Sieve", font_size=20, color="#CCCCCC")
        sieve_machine = VGroup(funnel_outline, sieve_label)
        sieve_label.next_to(funnel_outline, UP, buff=0.1)
        
        # Fix Issue 21: Shift machine area to 'C3', 'E5'
        self.place_in_area(sieve_machine, "C3", "E5", scale_factor=1.0)
        sieve_center = sieve_machine.get_center()
        
        self.play(Create(funnel_outline), Write(sieve_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Fix Issue 20: Update position to 'B4' and scale 0.8
        circle_16 = Circle(radius=0.4, color="#00FF00", fill_opacity=0.5)
        label_16 = Text("16", font_size=24, color=WHITE).move_to(circle_16.get_center())
        num_16 = VGroup(circle_16, label_16)
        
        self.place_at_grid(num_16, "B4", scale_factor=0.8)
        self.play(FadeIn(num_16))
        
        # Move into sieve mouth
        self.play(num_16.animate.move_to(sieve_center + UP * 0.5))
        
        # Scale the '16' circle down to 10% size after passing through.
        # Move to bottom of sieve area
        self.play(
            num_16.animate.scale(0.1).move_to(sieve_center + DOWN * 1.5),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Fix Issue 22: Update num_3 to start at B4 with scale 0.8
        circle_3 = Circle(radius=0.4, color="#FF0000", fill_opacity=0.5)
        label_3 = Text("3", font_size=24, color=WHITE).move_to(circle_3.get_center())
        num_3 = VGroup(circle_3, label_3)
        
        self.place_at_grid(num_3, "B4", scale_factor=0.8)
        self.play(FadeIn(num_3))
        
        # Move towards sieve center
        self.play(num_3.animate.move_to(sieve_center + UP * 0.5))
        
        # The '3' circle remains large and bounces off the sieve.
        # Bounces to B6
        self.play(
            num_3.animate.move_to(self.grid["B6"]),
            rate_func=rush_from,
            run_time=1.5
        )
        self.wait(2)
