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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "PDFs describe how continuous probability is distributed.",
            "Remember: height is density, but area is probability.",
            "No matter the shape, total area is one."
        ]
        self.setup_layout("Summary and Visual Wrap-up", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Morph a Sigma Σ #FF00FF into an Integral ∫ #FFFF00.
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        
        sigma = MathTex(r"\sum", color="#FF00FF")
        integral = MathTex(r"\int", color="#FFFF00")
        
        # Position in the center of the right side - fixing Issue 31 & 32 (consistent with axes)
        self.place_in_area(sigma, "C3", "E5", scale_factor=2.0)
        self.place_in_area(integral, "C3", "E5", scale_factor=2.0)
        
        self.play(Write(sigma))
        self.wait(1)
        self.play(ReplacementTransform(sigma, integral))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Toggle between different PDF shapes #00FFFF, #00FF00, #FFFFFF.
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Remove the integral to make room for axes
        self.play(FadeOut(integral))
        
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1, 0.5],
            x_length=4.5,
            y_length=3,
            axis_config={"include_tip": False, "include_numbers": False}
        )
        # Use lower area to leave room for the label later
        self.place_in_area(axes, "C3", "E5", scale_factor=0.9)
        
        # Normal PDF: Cyan #00FFFF
        normal_curve = axes.plot(
            lambda x: np.exp(-0.5 * (x / 0.8)**2) / (0.8 * np.sqrt(2 * np.pi)),
            color="#00FFFF"
        )
        
        # Uniform PDF: Green #00FF00
        uniform_curve = axes.plot(
            lambda x: 0.33 if -1.5 <= x <= 1.5 else 0,
            discontinuities=[-1.5, 1.5],
            dt=0.1,
            color="#00FF00"
        )
        
        # Exponential-like PDF: White #FFFFFF
        exponential_curve = axes.plot(
            lambda x: np.exp(-(x + 2)) if x >= -2 else 0,
            color="#FFFFFF"
        )
        
        # Show shapes
        self.play(Create(axes))
        self.play(Create(normal_curve))
        self.wait(0.5)
        self.play(ReplacementTransform(normal_curve, uniform_curve))
        self.wait(0.5)
        self.play(ReplacementTransform(uniform_curve, exponential_curve))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Flash text 'Total Area = 1' #FF0000 across all shapes.
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        
        area_label = Text("Total Area = 1", color="#FF0000")
        # Fixing Issue 30 - use area for multi-word label to prevent overflow
        self.place_in_area(area_label, "B3", "B5", scale_factor=0.8)
        
        # Create area fill for current shape (exponential)
        exp_area = axes.get_area(exponential_curve, color="#FF0000", opacity=0.3)
        
        # Show first fill and label
        self.play(FadeIn(area_label), FadeIn(exp_area))
        self.play(Indicate(area_label))
        self.wait(0.5)
        
        # Sequence showing red fill across shapes
        # Transition back to Normal
        new_normal_curve = axes.plot(
            lambda x: np.exp(-0.5 * (x / 0.8)**2) / (0.8 * np.sqrt(2 * np.pi)),
            color="#00FFFF"
        )
        norm_area = axes.get_area(new_normal_curve, color="#FF0000", opacity=0.3)
        
        self.play(
            ReplacementTransform(exponential_curve, new_normal_curve),
            ReplacementTransform(exp_area, norm_area),
            run_time=1.5
        )
        self.play(Indicate(area_label))
        
        # Transition back to Uniform
        new_uniform_curve = axes.plot(
            lambda x: 0.33 if -1.5 <= x <= 1.5 else 0,
            discontinuities=[-1.5, 1.5],
            dt=0.1,
            color="#00FF00"
        )
        unif_area = axes.get_area(new_uniform_curve, color="#FF0000", opacity=0.3)
        
        self.play(
            ReplacementTransform(new_normal_curve, new_uniform_curve),
            ReplacementTransform(norm_area, unif_area),
            run_time=1.5
        )
        self.play(Indicate(area_label))
        
        self.wait(2)
