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
        self.setup_layout("Mathematical Mechanics: Multiply and Sum", [
            "Let's multiply values element-wise.",
            "Now sum the results together.",
            "This value represents the center.",
            "Example: one times A plus D.",
            "The core convolution mechanics."
        ])
        
        # Grid visualizers (using local SVGs for assets where possible or basic Mobjects)
        input_grid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        filter_grid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg")
        pixel_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixel.svg")
        formula_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/formula.svg")

        # Grouping for visual consistency
        matrix_group = VGroup(input_grid, filter_grid).arrange(RIGHT, buff=0.5)
        self.place_in_area(matrix_group, 'B2', 'B4', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(input_grid), FadeIn(filter_grid))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # Highlight pixel pairs
        pixel_pair = VGroup(pixel_asset).set_color("#FFFF00")
        self.place_at_grid(pixel_pair, 'C3')
        self.play(Create(pixel_pair))

        # === Animation for Lecture Line 2 ===
        calc_text = MathTex(r"1 \cdot A + 0 \cdot B + 0 \cdot C + 1 \cdot D").scale(0.7)
        result_text = MathTex(r"= A + D").scale(0.7)
        ex_label = Text("Example: 1*A + 1*D", font_size=20)
        calculation_group = VGroup(calc_text, result_text, ex_label).arrange(DOWN)
        
        self.place_in_area(calculation_group, 'D3', 'F5', scale_factor=0.6)
        
        self.play(Write(calc_text))
        self.lecture[1].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(ReplacementTransform(calc_text.copy(), result_text))
        self.lecture[2].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(ex_label))
        self.lecture[3].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        summary = Text("Convolution Operation", font_size=20, color=YELLOW)
        self.place_at_grid(summary, 'A4', scale_factor=0.8)
        self.place_in_area(formula_asset, 'C1', 'C2', scale_factor=0.5)
        self.play(FadeIn(summary), FadeIn(formula_asset))
        self.lecture[4].set_color("#FF8800")
        self.wait(2)
