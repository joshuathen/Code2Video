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
        # Title and Lecture Lines
        title_text = "Prerequisite Checklist: Population vs. Sample"
        lecture_lines = [
            "The population includes every individual in a group.",
            "We select a sample to measure and label.",
            "Scaling up the sample allows for detailed analysis."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Assets and Colors
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/dots.svg]
        pop_dots = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/dots.svg")
        pop_dots.set_color(WHITE)
        
        pop_color = "#FFFFFF"
        sample_color = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Display a large cloud of small white (#FFFFFF) dots representing the population.
        self.lecture[0].set_color(YELLOW)
        
        # Position the dots in the main display area
        self.place_in_area(pop_dots, "B2", "E5", scale_factor=2.0)
        
        # Using Text instead of MathTex for stability
        pop_label = Text("Population (μ)", color=pop_color)
        # Fix: Issue 37 - Position pop_label at B3
        self.place_at_grid(pop_label, "B3", scale_factor=0.6)
        
        self.play(FadeIn(pop_dots), Write(pop_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw a green (#00FF00) circle around a group of dots and label them 'Population (μ)' and 'Sample (x̄)'.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Sample selection - highlighting a subset of the dots
        # We'll create a "sample" svg by copying the dots and coloring it green, then masking or positioning
        sample_dots = pop_dots.copy().set_color(sample_color)
        
        sample_center = self.grid["D4"]
        sample_circle = Circle(radius=0.7, color=sample_color, stroke_width=4)
        sample_circle.move_to(sample_center)
        
        # Label for Sample
        sample_label = Text("Sample (x̄)", color=sample_color)
        # Fix: Issue 38 - Position sample_label in area F3-F5
        self.place_in_area(sample_label, "F3", "F5", scale_factor=0.6)

        # We simulate the selection by showing the circle and perhaps "coloring" the dots within it
        # Since pop_dots is an SVG, we can't easily iterate over internal dots unless we decompose it.
        # But we can just layer a green circle and a smaller version of dots inside.
        
        self.play(Create(sample_circle))
        self.play(Write(sample_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Scale up the green circle until it fills the screen, showing the sampled dots clearly.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Fix: Issue 39 - Fade out pop_label before scaling
        self.play(FadeOut(pop_label))

        # Scaling and focus
        # We group sample elements to scale together. 
        # Note: In a real scenario, we might want to "clip" the dots to the circle, 
        # but here we'll scale the circle and the dots that look like they are inside.
        
        sample_group = VGroup(sample_circle, sample_label)
        
        # Zoom effect: move everything except the sample out
        self.play(
            FadeOut(pop_dots),
            sample_group.animate.scale(2.5).move_to(self.grid["C3"]),
            run_time=2
        )
        
        # Final state cleanup
        self.wait(2)
        self.lecture[2].set_color(WHITE)
        self.wait(1)
