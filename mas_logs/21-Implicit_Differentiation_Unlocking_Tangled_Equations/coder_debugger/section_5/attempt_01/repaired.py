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
        # Setup context
        title = "The Interaction: The Product Rule Trap"
        lines = [
            'Differentiating x times y requires the Product Rule.',
            'It becomes one times y plus x dy dx.',
            "Don't forget: y is a function, not a constant."
        ]
        self.setup_layout(title, lines)
        
        # Define Colors
        ORANGE_COLOR = "#FFA500"
        RED_COLOR = "#FF0000"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(ORANGE_COLOR))
        
        # Main Equation: x^2 + xy = 10
        eq_initial = VGroup(*[
            Text(t, font_size=40) for t in ["x²", "+", "xy", "=", "10"]
        ]).arrange(RIGHT, buff=0.15)
        self.place_in_area(eq_initial, "B2", "B5")
        self.play(Write(eq_initial))
        
        # Highlight xy
        highlight_box = SurroundingRectangle(eq_initial[2], color=ORANGE_COLOR, buff=0.1)
        self.play(Create(highlight_box))
        self.play(eq_initial[2].animate.set_color(ORANGE_COLOR))
        
        # Product Rule Helper
        product_rule = Text("(f · g)' = f'g + fg'", font_size=32, color=ORANGE_COLOR)
        self.place_in_area(product_rule, "C2", "C5")
        self.play(FadeIn(product_rule, shift=UP))
        self.wait(1.5)
        self.play(FadeOut(product_rule), Uncreate(highlight_box))

        # === Animation for Lecture Line 2 ===
        # Fixed syntax and completed differentiation step
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(ORANGE_COLOR)
        )
        
        # Differentiated Form: 2x + (1*y + x*dy/dx) = 0
        eq_diff = VGroup(*[
            Text(t, font_size=34) for t in ["2x", "+", "(1·y + x·dy/dx)", "=", "0"]
        ]).arrange(RIGHT, buff=0.15)
        self.place_in_area(eq_diff, "D1", "D6")
        
        self.play(Write(eq_diff))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(ORANGE_COLOR)
        )
        
        # Highlight the dy/dx part to emphasize y is a function
        caution_highlight = SurroundingRectangle(eq_diff[2], color=RED_COLOR, buff=0.1)
        self.play(Create(caution_highlight))
        self.play(eq_diff[2].animate.set_color(RED_COLOR))
        self.wait(2)