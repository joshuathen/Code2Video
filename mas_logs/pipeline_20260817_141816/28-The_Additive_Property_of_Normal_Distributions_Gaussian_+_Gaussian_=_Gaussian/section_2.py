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
        self.setup_layout("The Problem: Adding Two Random Variables", [
            "What happens when we add two independent variables?",
            "Imagine two distinct bell curves representing noise.",
            "They move toward each other to merge.",
            "This represents combined uncertainty in total time.",
            "Can we describe this sum as a new distribution?"
        ])

        # Define Gaussian function helper
        def gaussian(x, mu, sigma):
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        # Assets
        # Placeholder for [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # In a real scenario, this would be an SVGMobject or ImageMobject.
        # Since it's '/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg', we'll represent it with a subtle dot.
        asset_icon = Dot(color=WHITE, radius=0.05)
        
        axes = Axes(x_range=[-3, 3], y_range=[0, 0.5], axis_config={"include_numbers": False}).scale(0.4)
        
        curve1 = axes.plot(lambda x: gaussian(x, -0.75, 0.4), color=BLUE)
        curve2 = axes.plot(lambda x: gaussian(x, 0.75, 0.4), color=RED)
        
        label1 = Text("Variable X", color=BLUE, font_size=20)
        label2 = Text("Variable Y", color=RED, font_size=20)

        # Group components for positioning
        animation_group = VGroup(axes, curve1, curve2, asset_icon)
        
        # Apply layout constraints
        # 1. Place the main group in A4-F6 (VideoCritic #25, #26)
        self.place_in_area(animation_group, "A4", "F6", scale_factor=0.8)
        
        # 2. Place labels (VideoCritic #24)
        self.place_at_grid(label1, "B2", scale_factor=0.8)
        self.place_at_grid(label2, "B5", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Create(axes), Create(curve1), Create(curve2), Write(label1), Write(label2), FadeIn(asset_icon))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(curve1.animate.shift(RIGHT * 0.8), curve2.animate.shift(LEFT * 0.8))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.wait(2)
