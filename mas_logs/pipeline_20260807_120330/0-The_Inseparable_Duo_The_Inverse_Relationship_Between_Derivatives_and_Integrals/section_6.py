from manim import *
import numpy as np

# Set configuration to prevent FileNotFoundError during LaTeX cleanup
config.no_latex_cleanup = True

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
        # Setup the scene layout with title and lecture lines
        self.setup_layout("Application: The Power of the Relationship", [
            "This relationship lets us calculate areas with ease.",
            "Find the anti-derivative to solve complex integrals.",
            "Total distance is simply the anti-derivative of speed."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.lecture[0].set_color(YELLOW)
        
        # Define and place axes in the area C2 to F5 (Issue 40 fix)
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 5, 1],
            axis_config={"include_tip": True},
            x_length=3.5,
            y_length=3.5
        )
        self.place_in_area(axes, "C2", "F5")
        
        # Plot the parabola y=x^2 in green
        parabola = axes.plot(lambda x: x**2, x_range=[0, 2.2], color="#00FF00")
        parabola_label = MathTex(r"y=x^2", color="#00FF00")
        
        # Position label at grid C5 (Issue 41 fix)
        self.place_at_grid(parabola_label, "C5", scale_factor=0.8)
        
        # Shade the area from 0 to 2
        area = axes.get_area(parabola, x_range=[0, 2], color=YELLOW, opacity=0.3)
        
        self.play(Create(axes), Create(parabola))
        self.play(Write(parabola_label))
        self.play(FadeIn(area))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update lecture colors
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Display anti-derivative F(x) = x^3/3 and evaluation F(2) - F(0)
        anti_derivative = MathTex(r"F(x) = \frac{x^3}{3}", color=WHITE)
        self.place_at_grid(anti_derivative, "B2", scale_factor=0.8)
        
        evaluation = MathTex(r"F(2) - F(0)", color=WHITE)
        self.place_at_grid(evaluation, "B5", scale_factor=0.8)
        
        self.play(Write(anti_derivative))
        self.wait(0.5)
        self.play(Write(evaluation))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update lecture colors
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Replace the area label with the result "8/3" (#FFFF00) and flash it
        # Note: Storyboard says "Replace the area label with the result '8/3'".
        # In step 1 we didn't have an area label, just a curve label.
        # I'll create the result label and place it over the area.
        area_result = MathTex(r"\frac{8}{3}", color=YELLOW)
        # Position result near the center of the shaded area
        # Area is from x=0 to 2, y=0 to 4. Midpoint x=1, y=2 relative to axes.
        # Axes are in C2-F5.
        self.place_at_grid(area_result, "D3", scale_factor=0.9)
        
        self.play(FadeOut(evaluation))
        self.play(Write(area_result))
        self.play(Flash(area_result, color=YELLOW, flash_radius=0.5))
        self.wait(2)
