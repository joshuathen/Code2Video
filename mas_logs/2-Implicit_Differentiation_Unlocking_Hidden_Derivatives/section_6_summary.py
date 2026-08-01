from manim import *
import numpy as np

# Use Text as a fallback for MathTex to avoid FileNotFoundError when LaTeX is not installed
MathTex = Text

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

class Section6SummaryScene(TeachingScene):
    def construct(self):
        # Initialize layout with title and specific lecture lines
        self.setup_layout("Summary & Key Takeaway", [
            "Treat y as a function y(x) during differentiation.",
            "Always 'tag' derivatives of y with dy/dx.",
            "Now you can unlock the slope of any curve!"
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Display various implicit equations floating in dim white
        eq1 = MathTex("x^3 + y^3 = 3xy", color="#555555")
        eq2 = MathTex("x^2 + y^2 = 25", color="#555555")
        eq3 = MathTex(r"\frac{x^2}{4} + \frac{y^2}{9} = 1", color="#555555")
        eq4 = MathTex("y^2 = x^3 - x", color="#555555")
        
        self.place_at_grid(eq1, "A2", scale_factor=0.6)
        self.place_at_grid(eq2, "A5", scale_factor=0.6)
        self.place_at_grid(eq3, "E2", scale_factor=0.6)
        self.place_at_grid(eq4, "E5", scale_factor=0.6)
        
        # Centered text 'Treat y as y(x)' in bold white
        treat_y = Text("Treat y as y(x)", weight=BOLD, color="#FFFFFF")
        self.place_in_area(treat_y, "B2", "C5", scale_factor=0.8)
        
        self.play(
            FadeIn(eq1, eq2, eq3, eq4),
            Write(treat_y),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        
        # Large dy/dx symbol below the text pulsing in magenta
        dy_dx = MathTex(r"\frac{dy}{dx}", color="#FF00FF")
        self.place_in_area(dy_dx, "D2", "E5", scale_factor=1.5)
        
        self.play(FadeIn(dy_dx))
        # Pulsing effect
        self.play(dy_dx.animate.scale(1.2), run_time=0.6)
        self.play(dy_dx.animate.scale(1/1.2), run_time=0.6)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        
        # Final highlight of the universal method application
        self.play(
            Indicate(treat_y, color=WHITE),
            Flash(dy_dx, color="#FF00FF"),
            run_time=2
        )
        
        # Final pulse for all floating equations to signify they are solved
        self.play(
            *[Indicate(eq, color="#FFFFFF") for eq in [eq1, eq2, eq3, eq4]],
            run_time=2
        )
        self.wait(3)
