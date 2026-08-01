from manim import *
import numpy as np

# === Base Class (MUST NOT BE MODIFIED) ===
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

# === Section 3 Scene ===
class Section3Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title = "Prerequisite Review: The Integral as 'Accumulation'"
        lines = [
            "Integrals represent the accumulation of values over time.",
            "We calculate this as the area under a curve.",
            "This area sums up all the small changes."
        ]
        
        self.setup_layout(title, lines)

        # Colors from storyboard
        COLOR_WHITE = "#FFFFFF"
        COLOR_BLUE = "#0000FF"
        
        # === Animation for Lecture Line 1 ===
        # Line 1: Integrals represent the accumulation of values over time.
        # Highlights first lecture line
        self.play(self.lecture[0].animate.set_color(COLOR_WHITE))
        
        # Asset: Curve icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/curve.svg]
        # B034: Self-contained imports at top.
        curve_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/curve.svg").set_color(COLOR_WHITE)
        self.place_at_grid(curve_asset, 'A6', scale_factor=0.6)
        
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 4, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE}
        )
        # Fix Issue 35: Layout adjustment for axes to prevent crowding.
        self.place_in_area(axes, 'B1', 'F5', scale_factor=0.75)
        
        # Smooth curve representing a changing value (velocity)
        # B058: Use rate_functions.prefix if rate functions were used in play call.
        curve = axes.plot(
            lambda x: 0.1 * x**2 + 0.5 * np.sin(2 * x) + 1.2, 
            x_range=[0, 4.5], 
            color=COLOR_WHITE
        )
        
        # Fix Issues 36 & 37: Layout adjustment for labels to avoid overlap with axes.
        # B008: Replace MathTex with Text to ensure reliable rendering.
        x_label = Text("t", font_size=20, color=WHITE)
        y_label = Text("v(t)", font_size=20, color=WHITE)
        
        self.place_at_grid(x_label, 'F6', scale_factor=0.7)
        self.place_at_grid(y_label, 'A1', scale_factor=0.7)
        
        self.play(
            Create(axes), 
            Write(x_label), 
            Write(y_label),
            FadeIn(curve_asset),
            run_time=1.5
        )
        self.play(Create(curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: We calculate this as the area under a curve.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_BLUE)
        )
        
        # Create thin Riemann rectangles to represent discrete summation
        rects = axes.get_riemann_rectangles(
            curve,
            x_range=[0.5, 4.0],
            dx=0.15,
            color=COLOR_BLUE,
            fill_opacity=0.5,
            stroke_width=0.2
        )
        
        self.play(Create(rects, lag_ratio=0.05), run_time=2.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: This area sums up all the small changes.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_BLUE)
        )
        
        # Smooth area block representing the definite integral
        area = axes.get_area(
            curve,
            x_range=[0.5, 4.0],
            color=COLOR_BLUE,
            fill_opacity=0.8
        )
        
        # Morphing the rectangles into a solid area to show the limit of the sum
        self.play(ReplacementTransform(rects, area), run_time=2)
        self.wait(2)

        # Reset lecture colors to default for final shot
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
