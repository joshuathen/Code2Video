from manim import *

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
        self.setup_layout(
            "Conclusion: Independence vs. Bayesian Updating",
            [
                "Independence means new information doesn't change our beliefs.",
                "Dependence allows us to update certainty using Bayes' Theorem.",
                "Use these tools to master the logic of probability."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Color matching: Line 1 -> #D3D3D3
        self.lecture[0].set_color("#D3D3D3")
        
        # Probability Square components
        prob_square_container = VGroup()
        unit_square = Square(side_length=2, color="#D3D3D3", fill_opacity=0.1)
        
        # Vertical strip (Event A)
        event_a_strip = Rectangle(width=0.8, height=2.0, color="#ADD8E6", fill_opacity=0.5)
        event_a_strip.move_to(unit_square.get_left() + RIGHT * 0.4)
        
        # Horizontal strip (Event B)
        event_b_strip = Rectangle(width=2.0, height=0.8, color="#FFB6C1", fill_opacity=0.5)
        event_b_strip.move_to(unit_square.get_bottom() + UP * 0.4)
        
        # Intersection area
        intersection = Rectangle(width=0.8, height=0.8, color="#00FF00", fill_opacity=0.8)
        intersection.move_to(unit_square.get_left() + RIGHT * 0.4 + DOWN * 0.6)
        
        square_label = Text("Area = P(A) * P(B)", font_size=16, color="#D3D3D3")
        square_label.next_to(unit_square, DOWN, buff=0.2)
        
        prob_square_container.add(unit_square, event_a_strip, event_b_strip, intersection, square_label)
        self.place_in_area(prob_square_container, "A1", "C3", scale_factor=0.8)
        
        self.play(Create(prob_square_container))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color matching: Line 2 -> #FFB6C1
        self.lecture[1].set_color("#FFB6C1")
        
        # Shrinking Universe components
        shrinking_universe_container = VGroup()
        universe_rect = Rectangle(width=3, height=2, color="#D3D3D3", fill_opacity=0.1)
        event_b_region = Ellipse(width=1.5, height=1.2, color="#FFB6C1", fill_opacity=0.4)
        event_b_region.move_to(universe_rect.get_center() + RIGHT * 0.3)
        
        event_a_in_b = Circle(radius=0.3, color="#ADD8E6", fill_opacity=0.6)
        event_a_in_b.move_to(event_b_region.get_center() + LEFT * 0.2)
        
        universe_label = Text("Update based on evidence", font_size=16, color="#FFB6C1")
        universe_label.next_to(universe_rect, DOWN, buff=0.2)
        
        shrinking_universe_container.add(universe_rect, event_b_region, event_a_in_b, universe_label)
        self.place_in_area(shrinking_universe_container, "A4", "C6", scale_factor=0.8)
        
        self.play(Create(shrinking_universe_container))
        
        # Simulation of "shrinking" - highlight event B region
        self.play(
            universe_rect.animate.set_stroke(opacity=0.2).set_fill(opacity=0.05),
            event_b_region.animate.scale(1.1).set_stroke(width=4),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color matching: Line 3 -> #00FF00
        self.lecture[2].set_color("#00FF00")
        
        self.summary_label = Text("Master Your Logic", font_size=24, color="#FFFFFF")
        # Fix for Issue 31
        self.place_at_grid(self.summary_label, 'D1', scale_factor=0.8)
        
        self.bayes_formula = MathTex(
            r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}",
            color="#00FF00"
        )
        # Fix for Issue 30
        self.place_in_area(self.bayes_formula, 'D2', 'F6', scale_factor=1.0)
        
        # Transition: Fade out graphics, fade in summary and formula
        self.play(
            FadeOut(prob_square_container),
            FadeOut(shrinking_universe_container),
            Write(self.summary_label)
        )
        self.play(
            ReplacementTransform(self.summary_label, self.bayes_formula)
        )
        
        # Final Pulse of Bayes Theorem
        self.play(
            self.bayes_formula.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(2)
        
        # Final "Thank You" message
        thanks = Text("Thank You!", font_size=36, color=WHITE)
        # Position thank you in the center area of the right grid
        self.place_in_area(thanks, "B2", "E5", scale_factor=1.0)
        self.play(
            FadeOut(self.bayes_formula),
            Write(thanks)
        )
        self.wait(2)
