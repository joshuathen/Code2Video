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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("From Concrete to Abstract: The Prerequisite", [
            "Vectors are more than just arrows.",
            "We move beyond simple geometric vectors.",
            "A vector space is a playground.",
            "Two operations define this space: addition and scaling.",
            "These must satisfy eight formal axioms."
        ])
        
        # Load asset
        playground = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/playground.svg")
        self.place_in_area(playground, 'B2', 'E5', scale_factor=0.5)
        self.add(playground)
        
        # === Animation for Lecture Line 1 ===
        # An arrow vector (bright blue, #3498db) appears in the center.
        vec = Arrow(ORIGIN, RIGHT*1.5 + UP*1.0, color="#3498db")
        self.place_at_grid(vec, 'C2')
        self.play(Create(vec))
        self.lecture[0].set_color("#3498db")

        # === Animation for Lecture Line 2 ===
        # The arrow morphs into a smooth function curve (green, #2ecc71) while maintaining its origin point.
        curve = FunctionGraph(lambda x: 0.5 * np.sin(4 * x), x_range=[0, 1.5], color="#2ecc71")
        curve.move_to(vec.get_start(), aligned_edge=LEFT)
        self.play(Transform(vec, curve))
        self.lecture[1].set_color("#2ecc71")

        # === Animation for Lecture Line 3 ===
        # The text 'Vector Space' appears in white (#ffffff) at the top.
        vs_label = Text("Vector Space", color=WHITE)
        self.place_at_grid(vs_label, 'B3', scale_factor=0.9)
        self.play(Write(vs_label))
        self.lecture[2].set_color(WHITE)

        # === Animation for Lecture Line 4 ===
        # An 'Addition' symbol (+) and 'Scalar Multiplier' (c) appear, highlighted in yellow (#f1c40f).
        add_sym = MathTex("+", color="#f1c40f")
        scale_sym = MathTex("c", color="#f1c40f")
        group = VGroup(add_sym, scale_sym).arrange(RIGHT)
        self.place_at_grid(group, 'D3')
        self.play(FadeIn(add_sym), FadeIn(scale_sym))
        self.lecture[3].set_color("#f1c40f")

        # === Animation for Lecture Line 5 ===
        # The 8 axioms (represented as 8 floating white dots) orbit the space.
        dots = VGroup(*[Dot(color=WHITE, radius=0.05) for _ in range(8)])
        for dot in dots:
            self.place_at_grid(dot, 'D3', scale_factor=0.8)
        
        self.play(FadeIn(dots))
        self.lecture[4].set_color(WHITE)
        
        # Simple orbit animation
        self.play(Rotate(dots, angle=2*PI, about_point=playground.get_center(), run_time=3, rate_func=linear))
