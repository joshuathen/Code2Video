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
        # Fetching lecture lines from storyboard
        lecture_lines = [
            "A population represents the entire group we study.",
            "Samples are smaller subsets taken from that population.",
            "Distribution describes the shape and spread of data."
        ]
        
        self.setup_layout("Prerequisite Knowledge: Population vs. Sample", lecture_lines)
        
        # Asset path
        cookie_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cookie.svg"
        
        # === Animation for Lecture Line 1 ===
        # A large orange (#FF8C00) circle labeled 'Population' appears, filled with small cookie icons.
        self.lecture[0].set_color("#FF8C00")
        
        pop_circle = Circle(radius=1.8, color="#FF8C00", fill_opacity=0.1)
        # Scaling is handled by the radius here, but we'll use place_in_area to position it
        self.place_in_area(pop_circle, "A1", "D4")
        
        pop_label = Text("Population", color="#FF8C00", font_size=24)
        self.place_at_grid(pop_label, "A2", scale_factor=0.8) 
        
        # Cookies inside population - using Asset: cookie.svg
        cookies = VGroup()
        for _ in range(25):
            cookie = SVGMobject(cookie_asset).scale(0.15)
            # Apply color to the SVG mobject
            cookie.set_color("#D2691E")
            angle = np.random.uniform(0, 2 * PI)
            dist = np.random.uniform(0, 1.4)
            cookie_pos = pop_circle.get_center() + np.array([
                dist * np.cos(angle),
                dist * np.sin(angle),
                0
            ])
            cookie.move_to(cookie_pos)
            cookies.add(cookie)
        
        self.play(Create(pop_circle), Write(pop_label), FadeIn(cookies))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # A green (#00FF00) circle labeled 'Sample' appears as a small group of cookies moves into it.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00")
        
        sample_circle = Circle(radius=0.7, color="#00FF00", fill_opacity=0.1)
        self.place_in_area(sample_circle, "E5", "F6")
        
        sample_label = Text("Sample", color="#00FF00", font_size=20)
        self.place_at_grid(sample_label, "D5", scale_factor=0.8)
        
        # Select 5 cookies to move from population to sample
        sample_indices = np.random.choice(len(cookies), 5, replace=False)
        sample_cookies_subset = VGroup(*[cookies[i] for i in sample_indices])
        
        self.play(Create(sample_circle), Write(sample_label))
        
        # Move cookies to sample circle area
        move_animations = []
        for i, cookie in enumerate(sample_cookies_subset):
            angle = np.random.uniform(0, 2 * PI)
            dist = np.random.uniform(0, 0.4)
            target_pos = sample_circle.get_center() + np.array([
                dist * np.cos(angle),
                dist * np.sin(angle),
                0
            ])
            move_animations.append(cookie.animate.move_to(target_pos))
            
        self.play(*move_animations)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # White (#FFFFFF) labels 'Mean = mu' and 'Mean = x-bar' appear next to the circles.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)
        
        # Labels for means using MathTex for symbols
        # Population mean: mu
        mu_label = MathTex(r"\text{Mean } = \mu", color=WHITE, font_size=32)
        # Sample mean: x-bar
        xbar_label = MathTex(r"\text{Mean } = \bar{x}", color=WHITE, font_size=32)
        
        # Positioning: mu near population circle
        self.place_at_grid(mu_label, "C5", scale_factor=0.8)
        # Positioning: x-bar near sample circle (Issue 26: avoid overlap with sample_circle E5-F6)
        self.place_at_grid(xbar_label, "F4", scale_factor=0.8)
        
        self.play(FadeIn(mu_label), FadeIn(xbar_label))
        self.wait(3)
