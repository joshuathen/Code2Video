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
        self.setup_layout("The Change of Basis Formula", [
            "Define P, the transition matrix.", 
            "Coordinates in A equal P times B.", 
            "The grid warps to new coordinates.", 
            "P maps basis B to basis A.", 
            "Understand the shift with P."
        ])
        
        # Load asset
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")

        # === Animation for Lecture Line 1 ===
        # Display P = [b1 b2] as base change matrix
        p_matrix = MathTex(r"P = [b_1 \quad b_2]", font_size=36)
        self.place_at_grid(grid_asset, 'A2', scale_factor=0.5)
        self.place_at_grid(p_matrix, 'A4', scale_factor=1.0)
        self.play(Write(p_matrix), FadeIn(grid_asset))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Show X = P * X' formula
        formula = MathTex(r"[v]_A = P [v]_B", font_size=40)
        self.place_at_grid(formula, 'B4', scale_factor=0.9)
        self.play(Write(formula))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Color matrix P #FFFF00 for emphasis
        p_matrix.set_color(YELLOW)
        self.play(Indicate(p_matrix))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        # Animate X' vector scaling in the new basis
        vector = Arrow(ORIGIN, RIGHT + UP, color=RED)
        self.place_in_area(grid_asset, 'D4', 'F6', scale_factor=0.4)
        self.place_at_grid(vector, 'C4', scale_factor=0.7)
        self.play(GrowArrow(vector))
        self.lecture[3].set_color(YELLOW)

        # === Animation for Lecture Line 5 ===
        # Show resulting vector X in #00FFFF
        final_vector = Arrow(ORIGIN, RIGHT*0.5 + UP*1.5, color="#00FFFF")
        self.place_at_grid(final_vector, 'E4', scale_factor=0.7)
        self.play(Create(final_vector))
        self.lecture[4].set_color(YELLOW)
        self.wait(2)
