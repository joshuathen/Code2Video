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
        lecture_lines = ["Heat diffusion is inherently a smooth process.", "Higher frequencies dissipate heat much faster.", "Fourier series model these different decay rates."]
        self.setup_layout("Connecting to the Heat Equation", lecture_lines)
        
        # Assets
        rod = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg")
        self.place_in_area(rod, 'B3', 'E5', scale_factor=1.2)
        
        # Temperature distribution (initial sharp spike)
        def get_dist(t=0):
            x = np.linspace(-1, 1, 100)
            y = np.exp(-t * (x**2)) * np.exp(-abs(x * 5))
            return x, y

        x_coords, y_coords = get_dist(0)
        # Scale and position relative to rod, which is at the center of B3-E5
        center = self.grid['D4']
        coords = [np.array([x + center[0], y + center[1], 0]) for x, y in zip(x_coords, y_coords)]
        temp_curve = VMobject(color=WHITE)
        temp_curve.set_points_smoothly(coords)
        self.place_at_grid(temp_curve, 'D4', scale_factor=0.9)
        
        # Gradient indicator
        grad = Arrow(start=ORIGIN, end=RIGHT*0.5, color="#00FFFF")
        self.place_at_grid(grad, 'D2', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(rod), Create(temp_curve))
        self.lecture[0].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.play(Create(grad))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        def update_curve(mob, dt):
            # Slow evolution
            t = self.time * 0.5
            x, y = get_dist(t)
            center = self.grid['D4']
            new_coords = [np.array([val + center[0], y + center[1], 0]) for val, y in zip(x, y)]
            mob.set_points_smoothly(new_coords)
            
        temp_curve.add_updater(update_curve)
        self.wait(3)
        temp_curve.remove_updater(update_curve)
