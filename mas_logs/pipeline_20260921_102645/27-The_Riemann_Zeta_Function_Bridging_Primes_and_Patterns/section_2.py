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
        lecture_lines = [
            "Define zeta(s) as the infinite sum.",
            "Input complex numbers into the function.",
            "Observe outputs on the complex plane.",
            "Different s values map to unique points.",
            "These points form elegant geometric structures."
        ]
        self.setup_layout("Defining the Riemann Zeta Function", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        zeta_formula = MathTex(r"\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}", color=BLUE)
        self.place_in_area(zeta_formula, 'A2', 'C5', scale_factor=1.2)
        self.play(Write(zeta_formula))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        complex_input = MathTex(r"s = \sigma + it", color=YELLOW)
        self.place_at_grid(complex_input, 'D2', scale_factor=1.2)
        self.play(FadeIn(complex_input))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        complex_plane = ComplexPlane(x_range=[-2, 4], y_range=[-2, 2], axis_config={"include_numbers": False}).scale(0.5)
        self.place_in_area(complex_plane, 'D3', 'F5', scale_factor=0.9)
        self.place_in_area(grid_asset, 'D3', 'F5', scale_factor=0.5)
        self.play(FadeIn(grid_asset), Create(complex_plane))
        self.lecture[2].set_color(GREEN)

        # === Animation for Lecture Line 4 ===
        point = Dot(color=RED)
        point.move_to(complex_plane.c2p(2, 0))
        self.play(FadeIn(point))
        self.lecture[3].set_color(RED)

        # === Animation for Lecture Line 5 ===
        structure_label = Text("Geometric structures", font_size=20, color=PURPLE)
        self.place_at_grid(structure_label, 'F5', scale_factor=0.9)
        self.play(Write(structure_label))
        self.lecture[4].set_color(PURPLE)
        self.wait(2)
