from manim import *
import numpy as np
import random

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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Real-World Application: The Curse of Dimensionality", 
            [
                "Big data often lives in high-dimensional spaces.", 
                "Distance behaves differently in many-featured data sets.", 
                "Machine learning navigates these complex hypersphere clouds."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Use a distinct color for the active line
        self.lecture[0].set_color(YELLOW)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cloud.svg]
        cloud_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cloud.svg", color=BLUE_E, fill_opacity=0.3)
        
        # Create 100 points clumped together in a 2D circle
        dots = VGroup(*[Dot(radius=0.04, color="#58D68D") for _ in range(100)])
        
        # Random distribution in a circle
        random.seed(42) # For reproducibility
        for dot in dots:
            r = 0.5 * np.sqrt(random.random())
            theta = random.random() * 2 * PI
            dot.move_to([r * np.cos(theta), r * np.sin(theta), 0])
        
        # Group dots and cloud asset
        initial_group = VGroup(cloud_asset, dots)
        
        # Place in area (Fix for Issue #32: B4 to F6)
        self.place_in_area(initial_group, "B4", "F6", scale_factor=0.9)
        
        self.play(FadeIn(cloud_asset), FadeIn(dots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Fade out cloud asset as we enter higher dimensions
        self.play(FadeOut(cloud_asset))
        
        # Scatter widely across the grid (Avoiding Row A to protect title)
        tl = self.grid["B1"]
        br = self.grid["F6"]
        
        scatter_anims = []
        for dot in dots:
            target_x = random.uniform(tl[0], br[0])
            target_y = random.uniform(br[1], tl[1])
            scatter_anims.append(dot.animate.move_to([target_x, target_y, 0]))
            
        self.play(*scatter_anims, run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Pick one point to be the "High-Dimensional Data Point"
        special_dot = dots[42]
        
        # Create label
        label = Text("High-Dimensional Data Point", font_size=16, color="#FF0000")
        
        # Highlights
        self.play(
            special_dot.animate.set_color("#FF0000").scale(2.5),
            dots.animate.set_opacity(0.3)
        )
        special_dot.set_opacity(1.0)
        
        # Position label relative to the dot with a boundary check for the title
        if special_dot.get_center()[1] > 1.4:
            label.next_to(special_dot, DOWN, buff=0.1)
        else:
            label.next_to(special_dot, UP, buff=0.1)
            
        self.play(Write(label))
        
        # Pulse the special dot
        self.play(
            special_dot.animate.scale(1.4), 
            run_time=0.6, 
            rate_func=there_and_back
        )
        self.play(
            special_dot.animate.scale(1.4), 
            run_time=0.6, 
            rate_func=there_and_back
        )
        
        self.wait(3)
