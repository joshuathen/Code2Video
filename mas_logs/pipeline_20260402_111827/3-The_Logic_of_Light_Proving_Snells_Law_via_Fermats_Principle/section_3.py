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

class Section3Scene(TeachingScene):
    def construct(self):
        # Define lecture lines
        lecture_lines = [
            "- Let’s map the journey from point A to B.",
            "- Light crosses the interface at a variable point x.",
            "- Distances depend on heights a, b and width L.",
            "- Pythagoras gives the length of each path segment.",
            "- These segments relate to travel times T1 and T2."
        ]
        
        # Initialize layout
        self.setup_layout("Setting the Geometric Stage", lecture_lines)
        
        # Define Colors for consistency
        C1 = YELLOW
        C2 = BLUE
        C3 = GREEN
        C4 = RED
        C5 = PURPLE_A

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(C1))
        
        interface = Line(self.grid["D1"], self.grid["D6"], color=WHITE)
        
        dot_a = Dot(self.grid["B2"], color=C1)
        label_a = Text("A", color=C1, font_size=24).next_to(dot_a, UP, buff=0.1)
        
        dot_b = Dot(self.grid["E5"], color=C1)
        label_b = Text("B", color=C1, font_size=24).next_to(dot_b, DOWN, buff=0.1)
        
        self.play(Create(interface))
        self.play(FadeIn(dot_a, label_a), FadeIn(dot_b, label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(C2))
        
        dot_x = Dot(self.grid["D4"], color=C2)
        label_x = Text("x", color=C2, font_size=24).next_to(dot_x, DOWN, buff=0.1)
        
        self.play(FadeIn(dot_x, label_x))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(C3))
        
        dash_a = DashedLine(self.grid["B2"], self.grid["D2"], color=C3)
        dash_b = DashedLine(self.grid["E5"], self.grid["D5"], color=C3)
        label_a_val = Text("a", color=C3, font_size=24).next_to(dash_a, LEFT, buff=0.1)
        label_b_val = Text("b", color=C3, font_size=24).next_to(dash_b, RIGHT, buff=0.1)
        
        arrow_l = DoubleArrow(self.grid["D2"], self.grid["D5"], buff=0, color=C3, tip_length=0.1)
        arrow_l.shift(UP * 0.4)
        label_l = Text("L", color=C3, font_size=24).next_to(arrow_l, UP, buff=0.05)
        
        brace_x = BraceBetweenPoints(self.grid["D2"], self.grid["D4"], direction=UP, color=C2, buff=0.1)
        label_x_seg = Text("x", color=C2, font_size=24).next_to(brace_x, UP, buff=0.05)
        
        brace_lx = BraceBetweenPoints(self.grid["D4"], self.grid["D5"], direction=UP, color=C2, buff=0.1)
        label_lx_seg = Text("L-x", color=C2, font_size=24).next_to(brace_lx, UP, buff=0.05)
        
        self.play(Create(dash_a), Create(dash_b), FadeIn(label_a_val, label_b_val))
        self.play(Create(arrow_l), FadeIn(label_l))
        self.play(Create(brace_x), FadeIn(label_x_seg), Create(brace_lx), FadeIn(label_lx_seg))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Using Text with Unicode instead of MathTex to avoid LaTeX FileNotFoundError
        self.play(self.lecture[3].animate.set_color(C4))
        
        path1 = Line(self.grid["B2"], self.grid["D4"], color=C4)
        path2 = Line(self.grid["D4"], self.grid["E5"], color=C4)
        
        formula_d1 = Text("d₁ = √(a² + x²)", color=C4, font_size=24)
        formula_d2 = Text("d₂ = √(b² + (L-x)²)", color=C4, font_size=24)
        
        self.place_at_grid(formula_d1, "A4", scale_factor=0.8)
        self.place_at_grid(formula_d2, "A5", scale_factor=0.8)
        
        self.play(Create(path1), Create(path2))
        self.play(Write(formula_d1), Write(formula_d2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Using Text with Unicode for total travel time
        self.play(self.lecture[4].animate.set_color(C5))
        
        time_total = Text("T = d₁/v₁ + d₂/v₂", color=C5, font_size=28)
        self.place_at_grid(time_total, "A6", scale_factor=0.8)
        
        self.play(Write(time_total))
        self.wait(2)
