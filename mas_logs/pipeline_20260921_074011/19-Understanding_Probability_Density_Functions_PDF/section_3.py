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
        lecture_lines = ["Density is always non-negative.", "The total area must equal one.", "This defines a valid probability distribution."]
        self.setup_layout("PDF Core Properties", lecture_lines)
        
        # Define objects
        axes = Axes(x_range=[-1, 5, 1], y_range=[-0.5, 2, 0.5], axis_config={"include_tip": False}, x_length=4, y_length=3)
        curve = axes.plot(lambda x: 1.5 * np.exp(-(x - 2)**2), x_range=[0, 4], color=YELLOW)
        area = axes.get_area(curve, x_range=[0, 4], color=BLUE, opacity=0.3)
        
        # Using SVG placeholder as per requirements
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        header = Text("PDF Properties", font_size=24)
        
        grid_group = VGroup(axes, curve, area, icon)
        labels = VGroup(
            Text("f(x) >= 0", font_size=20, color=RED),
            Text("Total Area = 1", font_size=20, color=BLUE)
        )
        
        # Apply layout fixes
        self.place_at_grid(header, 'A3', scale_factor=0.9)
        self.place_in_area(grid_group, 'B2', 'F5', scale_factor=0.6)
        self.place_in_area(labels, 'B2', 'F5', scale_factor=0.5)
        
        # Position labels relative to the grid_group
        labels[0].next_to(curve, UP)
        labels[1].next_to(area, DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(RED), FadeIn(axes), Create(curve), Write(labels[0]), FadeIn(icon))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE), FadeIn(area), Write(labels[1]))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN), Indicate(area))
        self.wait(2)
