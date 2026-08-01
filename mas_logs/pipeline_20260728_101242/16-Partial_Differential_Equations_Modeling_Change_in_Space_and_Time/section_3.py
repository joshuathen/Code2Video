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
        title_text = "The Heat Equation: Modeling Diffusion"
        lecture_lines = [
            "The heat equation describes how temperature spreads through space.",
            "Time change depends on the curvature of spatial temperature.",
            "High curvature means heat flows quickly into colder regions.",
            "Over time, temperature gradients flatten toward a steady state.",
            "This parabolic behavior models diffusion in many physical systems."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        RECT_COLOR = "#0000FF"
        SOURCE_COLOR = "#FFA500"
        FORMULA_HIGHLIGHT = "#FFFF00"
        CURVE_COLOR = "#FF0000"
        ARROW_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Show a blue rectangle (#0000FF) with an orange circle (#FFA500) labeled 'Heat Source'.
        self.lecture[0].set_color(RECT_COLOR)
        
        # Fix Issue 29: Position ice_block at 'B3' to 'E6' to avoid left-side crowding
        self.ice_block = Rectangle(width=4.5, height=3.5, color=RECT_COLOR, fill_opacity=0.3)
        self.place_in_area(self.ice_block, 'B3', 'E6')
        
        # Fix Issue 30: Position heat_source at 'C4' to remain centered in ice_block
        self.heat_source = Circle(radius=0.3, color=SOURCE_COLOR, fill_opacity=0.8)
        self.place_at_grid(self.heat_source, 'C4')
        
        source_label = Text("Heat Source", font_size=16, color=SOURCE_COLOR)
        self.place_at_grid(source_label, 'B4') 
        
        self.play(FadeIn(self.ice_block), FadeIn(self.heat_source), Write(source_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display formula: du/dt = alpha * d2u/dx2. Highlight du/dt in yellow (#FFFF00).
        self.lecture[1].set_color(FORMULA_HIGHLIGHT)
        
        # Fix Issue 31: Position formula at 'A3' to 'A6' for layout balance
        self.formula = MathTex(
            r"\frac{\partial u}{\partial t}", "=", r"\alpha", r"\frac{\partial^2 u}{\partial x^2}",
            font_size=36
        )
        self.formula[0].set_color(FORMULA_HIGHLIGHT)
        self.place_in_area(self.formula, 'A3', 'A6')
        
        self.play(Write(self.formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Plot a sharp red bell curve (#FF0000) over the blue rectangle.
        self.lecture[2].set_color(CURVE_COLOR)
        
        axes = Axes(
            x_range=[-2.2, 2.2],
            y_range=[0, 1.2],
            x_length=4.5,
            y_length=3.0,
            axis_config={"include_tip": False, "include_ticks": False},
            tips=False
        ).move_to(self.ice_block.get_center())
        
        sigma_tracker = ValueTracker(0.2)
        amplitude_tracker = ValueTracker(1.0)
        
        # Use ValueTracker for flattening animation
        curve = always_redraw(lambda: axes.plot(
            lambda x: amplitude_tracker.get_value() * np.exp(-(x**2) / (2 * sigma_tracker.get_value()**2)),
            color=CURVE_COLOR
        ))
        
        self.play(Create(curve))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Draw white arrows (#FFFFFF) pointing from the peak outward.
        self.lecture[3].set_color(ARROW_COLOR)
        
        peak_pos = axes.c2p(0, amplitude_tracker.get_value())
        arrow_l = Arrow(start=peak_pos, end=axes.c2p(-1.2, 0.4), color=ARROW_COLOR, buff=0.1)
        arrow_r = Arrow(start=peak_pos, end=axes.c2p(1.2, 0.4), color=ARROW_COLOR, buff=0.1)
        
        self.play(GrowArrow(arrow_l), GrowArrow(arrow_r))
        self.wait(1)
        
        # Animate the bell curve (#FF0000) flattening into a horizontal line.
        self.play(
            sigma_tracker.animate.set_value(2.5),
            amplitude_tracker.animate.set_value(0.2),
            FadeOut(arrow_l),
            FadeOut(arrow_r),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This parabolic behavior models diffusion in many physical systems.
        self.lecture[4].set_color(WHITE)
        
        # Final flattening to nearly horizontal
        self.play(
            sigma_tracker.animate.set_value(10.0),
            amplitude_tracker.animate.set_value(0.05),
            run_time=2
        )
        self.wait(2)
