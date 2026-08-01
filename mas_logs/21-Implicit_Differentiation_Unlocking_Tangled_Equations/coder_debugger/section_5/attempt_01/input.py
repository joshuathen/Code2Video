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
            "The product xy requires special attention.",
            "Use the product rule: first times derivative second.",
            "Plus second times derivative of the first.",
            "Combine with the rest of the equation.",
            "Isolate dy/dx to find the derivative."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        ORANGE_COLOR = "#FFA500"
        RED_COLOR = "#FF0000"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(ORANGE_COLOR))
        
        # Main Equation: x^2 + xy = 10
        # Issue 43: Area B2-B6, scale 0.9
        eq_initial = VGroup(*[
            Text(t, font_size=40) for t in ["x²", "+", "xy", "=", "10"]
        ]).arrange(RIGHT, buff=0.2)
        self.place_in_area(eq_initial, "B2", "B6", scale_factor=0.9)
        self.play(Write(eq_initial))
        
        # Highlight xy
        highlight_box = SurroundingRectangle(eq_initial[2], color=ORANGE_COLOR, buff=0.1)
        self.play(Create(highlight_box))
        self.play(eq_initial[2].animate.set_color(ORANGE_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(ORANGE_COLOR)
        )
        
        # Issue 42: Product rule formula (D2-D5, scale 0.7)
        product_rule = Text("(f · g)' = f'g + f g'", font_size=32, color=ORANGE_COLOR)
        self.place_in_area(product_rule, "D2", "D5", scale_factor=0.7)
        self.play(FadeIn(product_rule, shift=UP))
        
        # Interaction component 1: derivative of the first (1) times the second (y)
        interaction_1 = Text("(1)(y)", font_size=36)
        self.place_at_grid(interaction_1, "C2")
        self.play(Write(interaction_1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(ORANGE_COLOR)
        )
        
        # Interaction component 2: first (x) times derivative of the second (dy/dx)
        interaction_2 = VGroup(
            Text("(x)(", font_size=36),
            Text("dy/dx", font_size=36, color=RED_COLOR),
            Text(")", font_size=36)
        ).arrange(RIGHT, buff=0.1)
        self.place_at_grid(interaction_2, "C5")
        self.play(Write(interaction_2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(ORANGE_COLOR)
        )
        
        # Full Equation: 2x + y + x(dy/dx) = 0
        # Issue 41: E2-E6, scale 0.8
        eq_diff = VGroup(*[
            Text(t, font_size=34) for t in ["2x", "+", "y", "+", "x(", "dy/dx", ")", "=", "0"]
        ]).arrange(RIGHT, buff=0.15)
        eq_diff[5].set_color(RED_COLOR)
        self.place_in_area(eq_diff, "E2", "E6", scale_factor=0.8)
        
        self.play(
            FadeOut(product_rule), 
            FadeOut(interaction_1), 
            FadeOut(interaction_2), 
            Uncreate(highlight_box),
            Write(eq_diff)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(ORANGE_COLOR)
        )
        
        # Isolation: x(dy/dx) = -2x - y
        eq_iso = VGroup(*[
            Text(t, font_size=34) for t in ["x(", "dy/dx", ")", "=", "-2x", "-", "y"]
        ]).arrange(RIGHT, buff=0.15)
        eq_iso[1].set_color(RED_COLOR)
        self.place_in_area(eq_iso, "F2", "F6", scale_factor=0.8)
        
        # Terms move to the right side
        self.play(ReplacementTransform(eq_diff.copy(), eq_iso))
        self.wait(2)
