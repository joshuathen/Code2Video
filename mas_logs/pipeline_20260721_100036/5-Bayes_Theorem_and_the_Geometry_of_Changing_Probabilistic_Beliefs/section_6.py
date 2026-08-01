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

class Section6Scene(TeachingScene):
    def construct(self):
        # setup_layout
        title = "Mapping Geometry to the Formula"
        lecture_lines = [
            "- Now let's map these shapes to the standard formula.",
            "- The numerator is the area of our target rectangle.",
            "- The denominator is the sum of all shaded areas.",
            "- Color-coded labels link the formula to the square's geometry.",
            "- Bayes' Theorem turns visual proportions into precise mathematical values."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_FORMULA = "#FFFFFF"
        COLOR_NUMERATOR = "#FFFF00" # Yellow
        COLOR_DENOMINATOR = "#00FFFF" # Cyan
        COLOR_RECT_1 = "#0000FF" # Blue
        COLOR_RECT_2 = "#00FF00" # Green

        # === Animation for Lecture Line 1 ===
        # Write P(H|E) = [P(H) * P(E|H)] / P(E) in white (#FFFFFF).
        # bayes_formula construction
        bayes_formula = MathTex(
            r"P(H|E)", r"=", r"{P(H) \cdot P(E|H)}", r"\over", r"{P(E)}",
            font_size=42, color=COLOR_FORMULA
        )
        
        # Issue 34: self.place_in_area(bayes_formula, 'C1', 'D6', scale_factor=0.8)
        self.place_in_area(bayes_formula, 'C1', 'D6', scale_factor=0.8)
        
        self.play(self.lecture[0].animate.set_color(COLOR_NUMERATOR))
        self.play(Write(bayes_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The numerator is the area of our target rectangle.
        target_rect = Rectangle(width=2, height=1.5, color=COLOR_RECT_1, fill_opacity=0.5)
        self.place_at_grid(target_rect, 'A3', scale_factor=0.8)
        
        # Issue 35: self.place_at_grid(h_highlight, 'C3', scale_factor=0.6)
        h_highlight = SurroundingRectangle(bayes_formula[2], color=COLOR_NUMERATOR)
        self.place_at_grid(h_highlight, 'C3', scale_factor=0.6)
        
        arrow1 = Arrow(start=h_highlight.get_top(), end=target_rect.get_bottom(), color=COLOR_NUMERATOR, buff=0.1)

        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_NUMERATOR)
        )
        self.play(FadeIn(target_rect))
        self.play(Create(h_highlight))
        self.play(GrowArrow(arrow1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The denominator is the sum of all shaded areas.
        other_rect = Rectangle(width=1.5, height=1.5, color=COLOR_RECT_2, fill_opacity=0.5)
        self.place_at_grid(other_rect, 'A5', scale_factor=0.8)
        
        # Issue 36: self.place_at_grid(denominator_highlight, 'D3', scale_factor=0.7)
        denominator_highlight = SurroundingRectangle(bayes_formula[4], color=COLOR_DENOMINATOR)
        self.place_at_grid(denominator_highlight, 'D3', scale_factor=0.7)
        
        sum_group = VGroup(target_rect, other_rect)
        arrow2 = Arrow(start=denominator_highlight.get_top(), end=sum_group.get_bottom(), color=COLOR_DENOMINATOR, buff=0.1)

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_DENOMINATOR)
        )
        self.play(FadeIn(other_rect))
        self.play(Create(denominator_highlight))
        self.play(GrowArrow(arrow2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Color-coded labels link the formula to the square's geometry.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_NUMERATOR)
        )
        label_num = Text("Numerator", font_size=18, color=COLOR_NUMERATOR)
        self.place_at_grid(label_num, 'B3', scale_factor=0.6)
        label_den = Text("Denominator", font_size=18, color=COLOR_DENOMINATOR)
        self.place_at_grid(label_den, 'B5', scale_factor=0.6)
        
        self.play(Write(label_num), Write(label_den))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Bayes' Theorem turns visual proportions into precise mathematical values.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(WHITE)
        )
        self.play(Indicate(bayes_formula, color=COLOR_FORMULA))
        self.wait(2)
