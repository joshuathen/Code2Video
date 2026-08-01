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
        # Initialize layout
        title_text = "Final Summary & Pro-Tips"
        lecture_lines = [
            "Differentiate every term, including constants which become zero.",
            "Always attach the dy over dx tag to y-derivatives.",
            "Solve for dy over dx using basic algebra.",
            "This technique is the foundation for related rates problems.",
            "Mastery of these steps makes complex calculus much easier."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        GREEN = "#00FF00"
        MAGENTA = "#FF00FF"
        CYAN = "#00FFFF"
        WHITE_COLOR = "#FFFFFF"

        # === Checklist Elements Preparation ===
        
        # Item 1: Differentiate
        box1 = Square(side_length=0.3, color=WHITE_COLOR)
        text1 = Text("Differentiate all terms (constants = 0)", font_size=18, color=WHITE_COLOR)
        item1 = VGroup(box1, text1).arrange(RIGHT, buff=0.2)
        self.place_in_area(item1, "A1", "A6")
        # Fixed check1: Replaced MathTex with Text to avoid LaTeX dependency error
        check1 = Text("✓", color=GREEN).scale(0.7).move_to(box1)

        # Item 2: Attach dy/dx
        box2 = Square(side_length=0.3, color=WHITE_COLOR)
        text2 = Text("Attach dy/dx to y-derivatives", font_size=18, color=WHITE_COLOR)
        item2 = VGroup(box2, text2).arrange(RIGHT, buff=0.2)
        self.place_in_area(item2, "B1", "B5")
        # Fixed check2 and dydx_symbol: Replaced MathTex with Text
        check2 = Text("✓", color=MAGENTA).scale(0.7).move_to(box2)
        dydx_symbol = Text("dy/dx", color=MAGENTA).scale(0.6)
        self.place_at_grid(dydx_symbol, "B6")

        # Item 3: Isolate
        box3 = Square(side_length=0.3, color=WHITE_COLOR)
        text3 = Text("Isolate dy/dx using algebra", font_size=18, color=WHITE_COLOR)
        item3 = VGroup(box3, text3).arrange(RIGHT, buff=0.2)
        self.place_in_area(item3, "C1", "C6")
        # Fixed check3: Replaced MathTex with Text
        check3 = Text("✓", color=CYAN).scale(0.7).move_to(box3)

        # Generic Formula - Fixed: Replaced MathTex with Text
        formula = Text("dy/dx = ...", color=CYAN).scale(1.0)
        formula_box = SurroundingRectangle(formula, color=CYAN, buff=0.2)
        formula_grp = VGroup(formula, formula_box)
        self.place_in_area(formula_grp, "D1", "D6")

        # Related Rates Connection
        rr_text = Text("Related Rates", color=MAGENTA, font_size=24)
        self.place_in_area(rr_text, "E3", "E6")
        arrow = Arrow(start=self.grid["C3"], end=self.grid["E2"], color=WHITE_COLOR, buff=0.1)

        # Mastery Title
        mastery = Text("Mastery Achieved", font_size=32, color=GREEN, weight=BOLD)
        self.place_in_area(mastery, "F1", "F6")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(GREEN))
        self.play(Create(box1), Write(text1))
        self.play(Write(check1), text1.animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(MAGENTA))
        self.play(Create(box2), Write(text2))
        self.play(Write(check2), text2.animate.set_color(MAGENTA))
        self.play(FadeIn(dydx_symbol))
        self.play(Indicate(dydx_symbol, scale_factor=1.4, color=MAGENTA))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(CYAN))
        self.play(Create(box3), Write(text3))
        self.play(Write(check3), text3.animate.set_color(CYAN))
        self.play(Create(formula_grp))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(MAGENTA))
        self.play(Write(rr_text), GrowArrow(arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(GREEN))
        checklist_visuals = VGroup(
            item1, check1, 
            item2, check2, dydx_symbol,
            item3, check3, formula_grp, 
            rr_text, arrow
        )
        self.play(checklist_visuals.animate.set_color(GREEN))
        self.play(Write(mastery))
        self.play(mastery.animate.scale(1.1), rate_func=there_and_back)
        self.wait(2)
