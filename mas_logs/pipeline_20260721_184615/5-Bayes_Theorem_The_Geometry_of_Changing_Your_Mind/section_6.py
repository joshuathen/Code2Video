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
        # Section title and lecture lines
        title_text = "The Formal Formula (The Mathematical Map)"
        lecture_lines = [
            "Let's map these areas to the algebraic formula.",
            "P(A) is the width of our initial belief.",
            "P(B given A) is the height of our slice.",
            "P(B) is the total area of the remaining rectangles.",
            "Bayes' Theorem simply calculates the ratio of these areas."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Hexadecimal colors as per L008
        BLUE_C = "#0000FF"
        GOLD_C = "#FFD700"
        WHITE_C = "#FFFFFF"
        GREEN_C = "#00FF00"
        RED_C = "#FF0000"
        GREY_C = "#555555"

        # === Animation for Lecture Line 1 ===
        # Display the Bayes' Theorem formula in the center (#FFFFFF).
        # We use double braces to allow reliable extraction of parts for coloring.
        formula = MathTex(
            "P(A|B) = { {{P(B|A)}} {{P(A)}} \\over {{P(B)}} }",
            color=WHITE_C
        )
        # Fix Issue 39: Scale factor 1.0 instead of 1.2
        self.place_in_area(formula, "A1", "A6", scale_factor=1.0)
        
        self.play(self.lecture[0].animate.set_color(WHITE_C))
        self.play(Write(formula))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Highlight P(A) and point to the initial blue width (#0000FF).
        
        # Background container (unit square context)
        # Using dimensions that will fit well within the grid
        rect_not_a = Rectangle(height=2.5, width=1.0, color=GREY_C, fill_opacity=0.1, stroke_width=1)
        # P(A) region (Blue column) - Width represents P(A)
        rect_a = Rectangle(height=2.5, width=1.5, color=BLUE_C, fill_opacity=0.2, stroke_width=2)
        rect_a.next_to(rect_not_a, RIGHT, buff=0)
        
        geometry_group = VGroup(rect_not_a, rect_a)
        # Fix Issue 40 & 41: Scale factor 0.7 and shift area to C1-F5
        self.place_in_area(geometry_group, "C1", "F5", scale_factor=0.7)
        
        # Width indicator for P(A)
        p_a_brace = Brace(rect_a, DOWN, color=BLUE_C)
        p_a_label = MathTex("P(A)", color=BLUE_C).scale(0.7) # L002 scale labels
        p_a_label.next_to(p_a_brace, DOWN, buff=0.1)
        
        self.play(self.lecture[1].animate.set_color(BLUE_C))
        # Highlight formula part
        p_a_part = formula.get_part_by_tex("P(A)")
        self.play(
            p_a_part.animate.set_color(BLUE_C),
            Indicate(p_a_part, color=BLUE_C) # L004 use Indicate
        )
        self.play(
            Create(rect_a),
            Create(rect_not_a),
            Create(p_a_brace),
            Write(p_a_label)
        )
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # Highlight P(B|A) and point to the slice height (#FFD700).
        
        # P(B|A) region (Green rectangle within A)
        rect_b_given_a = Rectangle(height=1.75, width=1.5, color=GREEN_C, fill_opacity=0.5, stroke_width=0)
        rect_b_given_a.move_to(rect_a.get_bottom(), aligned_edge=DOWN)
        
        # Height indicator for P(B|A)
        p_ba_brace = Brace(rect_b_given_a, RIGHT, color=GOLD_C)
        p_ba_label = MathTex("P(B|A)", color=GOLD_C).scale(0.7)
        p_ba_label.next_to(p_ba_brace, RIGHT, buff=0.1)
        
        self.play(self.lecture[2].animate.set_color(GOLD_C))
        # Highlight formula part
        p_ba_part = formula.get_part_by_tex("P(B|A)")
        self.play(
            p_ba_part.animate.set_color(GOLD_C),
            Indicate(p_ba_part, color=GOLD_C)
        )
        self.play(
            FadeIn(rect_b_given_a),
            Create(p_ba_brace),
            Write(p_ba_label)
        )
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # Highlight P(B) as the sum of Green and Red (#FFFFFF).
        
        # P(B|not A) region (Red rectangle within not A)
        rect_b_given_not_a = Rectangle(height=0.75, width=1.0, color=RED_C, fill_opacity=0.5, stroke_width=0)
        rect_b_given_not_a.move_to(rect_not_a.get_bottom(), aligned_edge=DOWN)
        
        # Total P(B) visual highlight (Outlines)
        p_b_outline = VGroup(
            rect_b_given_a.copy().set_fill(opacity=0).set_stroke(WHITE_C, 2),
            rect_b_given_not_a.copy().set_fill(opacity=0).set_stroke(WHITE_C, 2)
        )
        
        p_b_brace = Brace(geometry_group, UP, color=WHITE_C)
        p_b_label = MathTex("P(B)", color=WHITE_C).scale(0.7)
        p_b_label.next_to(p_b_brace, UP, buff=0.1)
        
        self.play(self.lecture[3].animate.set_color(WHITE_C))
        # Highlight formula part
        p_b_part = formula.get_part_by_tex("P(B)")
        self.play(
            p_b_part.animate.set_color(WHITE_C),
            Indicate(p_b_part, color=WHITE_C)
        )
        self.play(
            FadeIn(rect_b_given_not_a),
            Create(p_b_outline),
            Create(p_b_brace),
            Write(p_b_label)
        )
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # Box the entire formula to emphasize the final calculation (#FFFFFF).
        
        final_box = SurroundingRectangle(formula, color=WHITE_C, buff=0.2)
        
        self.play(self.lecture[4].animate.set_color(WHITE_C))
        self.play(Create(final_box))
        self.wait(2.0)
