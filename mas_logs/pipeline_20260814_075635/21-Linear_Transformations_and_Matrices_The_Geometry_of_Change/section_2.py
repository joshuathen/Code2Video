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
            "Basis vectors define the entire coordinate system.",
            "Observe i-hat at (1,0) and j-hat at (0,1).",
            "The transformation tracks where these basis vectors land."
        ]
        self.setup_layout("Basis Vectors: The Building Blocks", lecture_lines)
        
        # Setup grid and vectors
        axes = Axes(x_range=[-1, 3, 1], y_range=[-1, 3, 1], axis_config={"include_numbers": False}).scale(0.5)
        # Apply Fix from issue 28
        self.place_in_area(axes, 'B3', 'E5', scale_factor=0.6)
        
        # Assets (Using placeholders since the files were missing/not provided)
        icon_i = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        icon_j = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")

        i_hat = Vector(RIGHT, color=GREEN)
        j_hat = Vector(UP, color=GREEN)
        
        # Fix from issue 29
        self.place_at_grid(i_hat, 'D5', scale_factor=0.5)
        self.place_at_grid(j_hat, 'B3', scale_factor=0.5)

        i_hat_label = MathTex(r"\\hat{i}", color=GREEN).next_to(i_hat, DOWN)
        j_hat_label = MathTex(r"\\hat{j}", color=GREEN).next_to(j_hat, LEFT)
        
        basis_group = VGroup(i_hat, j_hat, i_hat_label, j_hat_label, icon_i, icon_j)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.play(Create(axes), Create(basis_group))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#0000FF"))
        grid = NumberPlane(x_range=[-1, 3, 1], y_range=[-1, 3, 1]).scale(0.5).move_to(axes.get_center())
        self.play(FadeIn(grid, run_time=1))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        
        # Transformation: i_hat to (2,0), j_hat to (0,2)
        new_i_hat = Vector(2*RIGHT, color=RED)
        new_j_hat = Vector(2*UP, color=RED)
        
        # Applying Fix 30 (Using suggested placement for new vectors)
        self.place_at_grid(new_i_hat, 'A3', scale_factor=0.7)
        self.place_at_grid(new_j_hat, 'A3', scale_factor=0.7)
        
        self.play(
            ReplacementTransform(i_hat, new_i_hat),
            ReplacementTransform(j_hat, new_j_hat),
            run_time=2
        )
        self.wait(1)
