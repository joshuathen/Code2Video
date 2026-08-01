from manim import *
import pathlib

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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        lecture_lines = [
            "The rate of area growth is the function's height.", 
            "This means the derivative of area is the function.", 
            "Integration and differentiation are inverse operations.", 
            "One builds the area, the other finds its slope.", 
            "This connection is the Fundamental Theorem of Calculus."
        ]
        self.setup_layout("The Fundamental Theorem: The 'Undo' Button", lecture_lines)
        
        # Define colors
        COLOR_EQ = "#FFFFFF"
        COLOR_HIGHLIGHT = "#FFFF00"
        COLOR_RESULT = "#00FF00"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        # Equation: A'(x) = d/dx [ ∫ f(t) dt ]
        # Using Unicode integral sign for environment safety
        eq_lhs = Text("A'(x) = d/dx [ \u222b f(t) dt ]", color=COLOR_EQ, font_size=32)
        self.place_in_area(eq_lhs, "B4", "C6", scale_factor=0.9)
        self.play(Write(eq_lhs))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_HIGHLIGHT))
        # Highlight the integral term and label it 'Area Function'
        box = SurroundingRectangle(eq_lhs, color=COLOR_HIGHLIGHT, buff=0.1)
        label = Text("Area Function", color=COLOR_HIGHLIGHT, font_size=18)
        label.next_to(box, UP, buff=0.1)
        self.play(Create(box), Write(label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_HIGHLIGHT))
        # Show the derivative operator applying and the box transforming into the result
        eq_rhs = Text("f(x)", color=COLOR_RESULT, font_size=40)
        self.place_in_area(eq_rhs, "D4", "E6", scale_factor=0.9)
        
        self.play(
            ReplacementTransform(box, eq_rhs),
            FadeOut(eq_lhs),
            FadeOut(label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_HIGHLIGHT))
        # Reveal the result f(x) with a flash effect
        self.play(Flash(eq_rhs, color=COLOR_RESULT, line_length=0.3, flash_radius=0.5))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(COLOR_HIGHLIGHT))
        # Summary text with SVG asset
        ftc_label = Text("Differentiation and Integration are Inverses", color=COLOR_EQ, font_size=20)
        self.place_in_area(ftc_label, "F4", "F6", scale_factor=0.8)
        
        # Load asset
        button_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/button.svg")
        button_icon.scale(0.3)
        button_icon.next_to(ftc_label, LEFT, buff=0.2)
        
        self.play(
            FadeIn(ftc_label),
            FadeIn(button_icon)
        )
        self.wait(3)
