from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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
        # Set random seed for consistent dust generation
        np.random.seed(42)
        
        self.setup_layout("The Master Catalog: The Mandelbrot Set", [
            "The Mandelbrot Set catalogs all possible Julia Sets.",
            "It depends on the starting parameter value c.",
            "Inside, the Julia Set forms one connected piece.",
            "Outside, it shatters into a dust of points.",
            "It is a dictionary of infinite dynamical worlds."
        ])
        
        # Colors
        COLOR_MANDELBROT = "#90EE90" # Light Green
        COLOR_POINT = "#FFFF00"      # Yellow
        COLOR_JULIA = "#ADD8E6"      # Light Blue
        COLOR_DUST = "#F08080"       # Light Coral

        # === Animation for Lecture Line 1 ===
        # The Mandelbrot Set catalogs all possible Julia Sets.
        self.lecture[0].set_color(COLOR_MANDELBROT)
        
        # Stylized Mandelbrot silhouette using a Cardioid and a Bulb
        mandel_cardioid = ParametricFunction(
            lambda t: np.array([
                0.25 * (2 * np.cos(t) - np.cos(2 * t)),
                0.25 * (2 * np.sin(t) - np.sin(2 * t)),
                0
            ]),
            t_range=[0, 2*PI],
            color=COLOR_MANDELBROT,
            fill_opacity=0.6
        ).scale(2.8).shift(LEFT*0.3)
        
        mandel_bulb = Circle(radius=0.5, color=COLOR_MANDELBROT, fill_opacity=0.6).next_to(mandel_cardioid, LEFT, buff=-0.1)
        mandel_silhouette = VGroup(mandel_cardioid, mandel_bulb)
        
        # Position the Mandelbrot set in the center-right area
        # Fix (Issue 36): scale_factor adjusted from 0.85 to 0.7
        self.place_in_area(mandel_silhouette, 'B2', 'E5', scale_factor=0.7)
        
        self.play(FadeIn(mandel_silhouette))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It depends on the starting parameter value c.
        self.lecture[1].set_color(COLOR_POINT)
        
        dot_c = Dot(color=COLOR_POINT)
        point_c_label = MathTex("c", color=COLOR_POINT, font_size=28)
        
        # Place c inside the main cardioid
        self.place_at_grid(dot_c, 'C3')
        point_c_label.next_to(dot_c, UP, buff=0.1)
        
        self.play(FadeIn(dot_c), Write(point_c_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Inside, the Julia Set forms one connected piece.
        self.lecture[2].set_color(COLOR_JULIA)
        
        # Asset (Issue 20): Use snowflake.svg
        snowflake = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/snowflake.svg", color=COLOR_JULIA)
        snowflake.set_fill(COLOR_JULIA, opacity=0.4)
        julia_label_1 = Text("Connected Julia", font_size=18, color=COLOR_JULIA)
        
        # Fix (Issue 34): Move snowflake to E6 to avoid overlap
        self.place_at_grid(snowflake, 'E6', scale_factor=0.8)
        julia_label_1.next_to(snowflake, DOWN, buff=0.2)
        
        self.play(DrawBorderThenFill(snowflake), FadeIn(julia_label_1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Outside, it shatters into a dust of points.
        self.lecture[3].set_color(COLOR_DUST)
        
        dot_outside = Dot(color=COLOR_POINT)
        point_c_label_out = MathTex("c", color=COLOR_POINT, font_size=28)
        
        # Fix (Issue 35): Move dot_outside to A4 to avoid clutter on the left
        self.place_at_grid(dot_outside, 'A4', scale_factor=0.8)
        point_c_label_out.next_to(dot_outside, UP, buff=0.1)
        
        # "Dust" Julia Set: scattered tiny dots
        # Positioning dust relative to its c point
        dust_center = self.grid['B5'] # Moving the dust away from the point for clarity
        dust_dots = VGroup(*[
            Dot(radius=0.02, color=COLOR_DUST).move_to(
                dust_center + np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 0])
            ) for _ in range(50)
        ])
        julia_label_2 = Text("Dust Julia", font_size=18, color=COLOR_DUST)
        julia_label_2.next_to(dust_dots, DOWN, buff=0.2)
        
        self.play(FadeIn(dot_outside), Write(point_c_label_out))
        self.play(FadeIn(dust_dots), FadeIn(julia_label_2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # It is a dictionary of infinite dynamical worlds.
        self.lecture[4].set_color(WHITE)
        
        # Dictionary visual: several small fractal-like icons appearing
        # Avoiding already used grid positions: E6, A4, B5, C3, B2-E5 area center
        icons = VGroup()
        positions = ['B6', 'D1', 'E2', 'A6', 'F4']
        for pos in positions:
            icon = Star(n=6, outer_radius=0.15, inner_radius=0.07, color=WHITE, fill_opacity=0.3)
            self.place_at_grid(icon, pos)
            icons.add(icon)
            
        self.play(LaggedStart(*[FadeIn(icon) for icon in icons], lag_ratio=0.3))
        self.wait(2)
