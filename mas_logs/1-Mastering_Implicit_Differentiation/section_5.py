from manim import *; MathTex = Text
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
        # Initial layout with provided title and lecture lines
        self.setup_layout("Complex Application: The Folium of Descartes", [
            "Complex loops like this are hard to solve for y.",
            "Implicit differentiation makes finding the slope much easier.",
            "Use the product rule on terms like six x y.",
            "Six x y becomes six y plus six x dy dx.",
            "Combine these rules to master any complex curve."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.lecture[0].set_color(YELLOW)
        
        # Draw Folium curve (Purple #BF00FF) - Parametric loop: x^3 + y^3 = 6xy
        # Parametric equations: x = 6t/(1+t^3), y = 6t^2/(1+t^3)
        # The loop forms for t in [0, inf). Using [0, 50] for visual completeness.
        folium = ParametricFunction(
            lambda t: np.array([
                (6 * t) / (1 + t**3),
                (6 * t**2) / (1 + t**3),
                0
            ]),
            t_range=[0, 50],
            color="#BF00FF"
        )
        # Position curve in bottom-right area using grid constraints
        self.place_in_area(folium, "D2", "F6", scale_factor=0.55)
        
        # Display initial equation 'x^3 + y^3 = 6xy' at the top of the right panel
        eq_main = MathTex(r"x^3 + y^3 = 6xy", color=WHITE)
        self.place_in_area(eq_main, "A1", "A6", scale_factor=1.0)
        
        self.play(Create(folium), Write(eq_main))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition lecture highlight
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Step: Apply differentiation operator to both sides
        eq_diff_step = MathTex(r"\frac{d}{dx}(x^3 + y^3) = \frac{d}{dx}(6xy)", color=WHITE)
        self.place_in_area(eq_diff_step, "A1", "A6", scale_factor=0.9)
        self.play(Transform(eq_main, eq_diff_step))
        self.wait(1)
        
        # Perform differentiation on the LHS (Implicit rule: dy/dx for y terms)
        eq_lhs_done = MathTex(r"3x^2 + 3y^2 \frac{dy}{dx} = \frac{d}{dx}(6xy)", color=WHITE)
        self.place_in_area(eq_lhs_done, "A1", "A6", scale_factor=0.9)
        self.play(Transform(eq_main, eq_lhs_done))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition lecture highlight
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Indicate product rule needed for '6xy'
        # Position label near the RHS term
        rule_label = Text("Product Rule", font_size=20, color="#BF00FF")
        self.place_at_grid(rule_label, "B5", scale_factor=1.0)
        
        self.play(Write(rule_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Transition lecture highlight
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Transform right side to '6y + 6x * dy/dx' using the product rule
        eq_rhs_done = MathTex(r"3x^2 + 3y^2 \frac{dy}{dx} = 6y + 6x \frac{dy}{dx}", color=WHITE)
        self.place_in_area(eq_rhs_done, "A1", "A6", scale_factor=0.85)
        self.play(Transform(eq_main, eq_rhs_done), FadeOut(rule_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Transition lecture highlight
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Rearrange: Move all terms with 'dy/dx' to the left side
        eq_rearranged = MathTex(r"3y^2 \frac{dy}{dx} - 6x \frac{dy}{dx} = 6y - 3x^2", color=WHITE)
        self.place_in_area(eq_rearranged, "B1", "B6", scale_factor=0.85)
        self.play(Write(eq_rearranged))
        self.wait(1)
        
        # Final display: Factor out 'dy/dx' and show the isolated result
        eq_final = MathTex(r"\frac{dy}{dx} = \frac{6y - 3x^2}{3y^2 - 6x}", color=WHITE)
        self.place_in_area(eq_final, "C1", "C6", scale_factor=1.0)
        self.play(Write(eq_final))
        self.wait(2)
        
        # Final cleanup: Return last lecture line to white
        self.lecture[4].set_color(WHITE)
        self.wait(2)
