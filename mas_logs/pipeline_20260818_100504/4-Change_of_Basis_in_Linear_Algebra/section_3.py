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
        lecture_lines = [
            "Coordinates in basis B are [v]B.",
            "The formula [v]std = P[v]B is the bridge.",
            "Matrix P represents the transformation mechanism.",
            "Numerical representations change as the grid shifts.",
            "The underlying vector remains invariant."
        ]
        self.setup_layout("The Transformation Mechanics", lecture_lines)
        
        # Assets
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_asset, 'A1', 'F6', scale_factor=2.0)
        grid_asset.set_opacity(0.3)
        self.add(grid_asset)
        
        # Elements
        vec_p = Vector([1, 2], color=WHITE)
        self.place_at_grid(vec_p, 'C2', scale_factor=0.7)
        label_p = MathTex(r"v", color=WHITE).next_to(vec_p.get_end(), UP)
        
        formula = MathTex(r"[v]_{std} = P[v]_B", color=YELLOW)
        self.place_in_area(formula, 'A4', 'B6', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(vec_p), Write(label_p))
        self.play(self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        self.play(Write(formula))
        self.play(self.lecture[1].animate.set_color(YELLOW))

        # === Animation for Lecture Line 3 ===
        vec_transformed = Vector([2, 1], color=PURPLE)
        self.place_at_grid(vec_transformed, 'E4', scale_factor=0.7)
        label_transformed = MathTex(r"v_B", color=PURPLE).next_to(vec_transformed.get_end(), DOWN)
        self.play(Create(vec_transformed), Write(label_transformed))
        self.play(self.lecture[2].animate.set_color(PURPLE))

        # === Animation for Lecture Line 4 ===
        rect = SurroundingRectangle(formula, color="#00FFFF", buff=0.1)
        self.play(Create(rect))
        self.play(self.lecture[3].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 5 ===
        self.play(FadeOut(vec_p), FadeOut(label_p))
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
