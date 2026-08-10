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
            "Replace a column with the target vector b.",
            "New parallelograms emerge from this replacement.",
            "Area ratios reveal the needed scale factors.",
            "The ratio gives x for the first column.",
            "This process defines Cramer's elegant rule."
        ]
        self.setup_layout("Geometric Substitution", lecture_lines)
        
        # Asset paths
        parallelogram_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg"

        # Grid object
        grid = NumberPlane(x_range=[-1, 5], y_range=[-1, 5], axis_config={"color": "#808080"}).scale(0.4)
        self.place_in_area(grid, "A1", "F6")

        # Mobjects
        para_orig = SVGMobject(parallelogram_asset, color=WHITE)
        para_new = SVGMobject(parallelogram_asset, color=RED)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#0000FF")
        self.play(FadeIn(grid))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        self.place_in_area(para_orig, 'B2', 'D4', scale_factor=0.8)
        self.play(Create(para_orig))
        
        # New parallelogram (swap)
        self.place_in_area(para_new, 'B2', 'D4', scale_factor=0.8)
        self.play(Transform(para_orig, para_new))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        area_label = MathTex(r"\text{Ratio} = \frac{\text{Det}(A_x)}{\text{Det}(A)}", color="#FFD700")
        self.place_at_grid(area_label, 'D3', scale_factor=0.9)
        self.play(FadeIn(area_label))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#0000FF")
        x_val = MathTex(r"x_1 = \frac{\text{Det}(A_x)}{\text{Det}(A)}", color="#0000FF")
        self.place_at_grid(x_val, 'D5', scale_factor=0.9)
        self.play(Write(x_val))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#800080")
        self.play(Indicate(self.lecture[4]))
