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

class Section6Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "We use samples to predict behavior of entire populations.",
            "This is vital for medical trials and quality control.",
            "The bell curve brings order to our chaotic world."
        ]
        self.setup_layout("Real-World Application: Why It Matters", lecture_lines)
        
        # Color definitions
        TEAL = "#008080"
        PURPLE = "#800080"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # A grid of product icons [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/product.svg] appears in teal (#008080).
        self.lecture[0].set_color(TEAL)
        
        # Load asset
        product_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/product.svg")
        product_icon.set_color(TEAL)
        
        # Create a grid of icons
        icons = VGroup(*[
            product_icon.copy().scale(0.25)
            for _ in range(30)
        ]).arrange_in_grid(rows=5, cols=6, buff=0.3)
        
        # Position the grid in a smaller area to avoid clutter
        self.place_in_area(icons, 'B2', 'E5', scale_factor=0.6)
        
        self.play(LaggedStart(*[FadeIn(icon) for icon in icons], lag_ratio=0.05), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A purple (#800080) bell curve is overlaid on the grid to show quality distribution.
        self.lecture[1].set_color(PURPLE)
        
        # Setup invisible axes for the bell curve
        axes = Axes(
            x_range=[-3, 3],
            y_range=[0, 4],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": False}
        )
        # Position axes with scaling for breathing room
        self.place_in_area(axes, 'A1', 'F6', scale_factor=0.85)
        
        # Gaussian function
        def bell_curve_func(x):
            return 3.5 * np.exp(-x**2 / (2 * 0.9**2))
            
        curve = axes.plot(bell_curve_func, color=PURPLE, stroke_width=6)
        
        # Overlap the curve on top of the icons
        self.play(Create(curve), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The center region of the bell curve is highlighted in white (#FFFFFF) to represent the target range.
        self.lecture[2].set_color(WHITE_COLOR)
        
        # Create a segment of the curve and a filled area for the target range
        target_range = [-0.9, 0.9]
        
        highlighted_segment = axes.plot(
            bell_curve_func, 
            x_range=target_range, 
            color=WHITE_COLOR, 
            stroke_width=8
        )
        
        area_highlight = axes.get_area(
            curve, 
            x_range=target_range, 
            color=WHITE_COLOR, 
            opacity=0.4
        )
        
        # Highlight the center region
        self.play(
            Create(highlighted_segment),
            FadeIn(area_highlight),
            run_time=1.5
        )
        self.wait(3)
