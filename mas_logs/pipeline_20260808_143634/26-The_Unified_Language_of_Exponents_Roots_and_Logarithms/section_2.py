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
        self.setup_layout("Roots: The Inverse of the Exponent", [
            "Roots reverse the power operation.",
            "We find the base when growth is known.",
            "Calculate base from volume and time."
        ])
        
        # Load assets
        container = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/container.svg")
        cube_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        
        y_val = MathTex("y", color="#00FF00", font_size=48)
        radical = MathTex(r"\sqrt[\cdot]{\cdot}", color=WHITE, font_size=60)
        
        # === Animation for Lecture Line 1 ===
        # Display y and radical with container
        group1 = VGroup(container, y_val, radical).arrange(RIGHT)
        self.place_in_area(group1, 'A2', 'C4', scale_factor=0.8)
        self.play(FadeIn(group1))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show root operation extracting base b
        self.lecture[1].set_color(YELLOW)
        b_val = MathTex("b", color=WHITE, font_size=48)
        # Place formula root in D2-D4
        formula_root = MathTex(r"b = \sqrt[x]{y}", color=WHITE, font_size=48)
        self.place_in_area(formula_root, 'D2', 'D4', scale_factor=0.7)
        self.play(Write(formula_root))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Calculate base from volume and time
        self.lecture[2].set_color(YELLOW)
        growth_formula = MathTex(r"b = y^{1/x}", color=WHITE, font_size=48)
        self.place_at_grid(growth_formula, 'E3', scale_factor=0.6)
        
        # Include cube asset
        self.place_at_grid(cube_asset, 'B5', scale_factor=0.8)
        self.play(FadeIn(cube_asset), Write(growth_formula))
        self.wait(2)
