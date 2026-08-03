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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout
        self.setup_layout("The Shift: From 'What it is' to 'What it does'", [
            "- Mathematicians define vectors by behavior, not just appearance.",
            "- A vector space is any set following eight rules.",
            "- If objects follow axioms, they act like vectors."
        ])
        
        # Colors for highlights
        H_COLOR = "#FFD700" # Gold color for symbols and rules
        
        # === Animation for Lecture Line 1 ===
        # Line: Mathematicians define vectors by behavior, not just appearance.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Factory icon creation (#FFFFFF)
        # Issue 27: Scale factory to 1.0 in area C3-D4
        factory_base = Rectangle(width=2.5, height=1.8, color=WHITE)
        chimney1 = Rectangle(width=0.4, height=0.7, color=WHITE).next_to(factory_base, UP, buff=0, aligned_edge=LEFT).shift(RIGHT*0.3)
        chimney2 = Rectangle(width=0.4, height=0.7, color=WHITE).next_to(factory_base, UP, buff=0, aligned_edge=RIGHT).shift(LEFT*0.3)
        factory = VGroup(factory_base, chimney1, chimney2)
        self.place_in_area(factory, 'C3', 'D4', scale_factor=1.0)
        
        # Input shapes: Circle, Square, Arrow
        circle = Circle(radius=0.35, color=BLUE)
        square = Square(side_length=0.7, color=RED)
        arrow = Arrow(start=LEFT*0.3, end=RIGHT*0.3, color=GREEN, buff=0)
        
        # Issue 26: Place shapes at Column 2 (C2, D2, E2)
        self.place_at_grid(circle, 'C2')
        self.place_at_grid(square, 'D2')
        self.place_at_grid(arrow, 'E2')
        
        self.play(
            Create(factory),
            FadeIn(circle, shift=RIGHT),
            FadeIn(square, shift=RIGHT),
            FadeIn(arrow, shift=RIGHT)
        )
        self.wait(0.5)
        
        # Shapes entering factory
        target_center = factory.get_center()
        self.play(
            circle.animate.move_to(target_center).set_opacity(0),
            square.animate.move_to(target_center).set_opacity(0),
            arrow.animate.move_to(target_center).set_opacity(0),
            run_time=1.5
        )

        # === Animation for Lecture Line 2 ===
        # Line: A vector space is any set following eight rules.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(H_COLOR)
        )
        
        # Inside symbols glowing in #FFD700
        # Use MathTex for symbols but ensure they are only created once (here).
        plus = MathTex("+", color=H_COLOR).scale(1.8)
        dot = MathTex("\\cdot", color=H_COLOR).scale(2.5)
        symbols = VGroup(plus, dot).arrange(RIGHT, buff=0.6).move_to(target_center)
        
        self.play(FadeIn(symbols))
        self.play(Indicate(symbols, color=H_COLOR, scale_factor=1.3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: If objects follow axioms, they act like vectors.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # "Follow the Rules" text above factory
        rules_label = Text("Follow the Rules", font_size=28, color=WHITE)
        self.place_in_area(rules_label, 'B3', 'B4', scale_factor=1.0)
        
        # Output shapes exit
        circle_out = Circle(radius=0.35, color=BLUE).move_to(target_center)
        square_out = Square(side_length=0.7, color=RED).move_to(target_center)
        arrow_out = Arrow(start=LEFT*0.3, end=RIGHT*0.3, color=GREEN, buff=0).move_to(target_center)
        
        self.play(Write(rules_label))
        
        # Exit to right side
        self.play(
            circle_out.animate.move_to(self.grid['C6']),
            square_out.animate.move_to(self.grid['D6']),
            arrow_out.animate.move_to(self.grid['E6']),
            run_time=1.5
        )
        self.wait(2)
        
        # Final cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
