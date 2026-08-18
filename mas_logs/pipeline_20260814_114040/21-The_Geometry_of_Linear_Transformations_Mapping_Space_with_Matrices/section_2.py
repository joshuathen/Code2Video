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
        self.setup_layout("The Basis Vectors: The DNA of Transformation", [
            "The transformation depends entirely on two basis vectors.",
            "The i-hat vector defines the grid's horizontal shift.",
            "The j-hat vector defines the grid's vertical shift."
        ])
        
        # Setup vectors
        i_hat = Vector(RIGHT, color=WHITE)
        j_hat = Vector(UP, color=RED)
        label_basis = Text("Basis", font_size=24, color=WHITE)
        
        # Load asset
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        
        group = VGroup(i_hat, j_hat)
        self.place_in_area(group, "A1", "B3", scale_factor=0.6)
        self.place_at_grid(label_basis, "B6", scale_factor=0.8)
        self.play(Create(i_hat), Create(j_hat), Write(label_basis))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        label_scaling = Text("Scaling", font_size=24, color=WHITE)
        self.place_at_grid(label_scaling, "C6", scale_factor=0.8)
        
        self.play(
            i_hat.animate.scale(2.0, about_point=ORIGIN), 
            Write(label_scaling)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#CCCCCC")
        label_grid = Text("Grid", font_size=24, color=WHITE)
        self.place_at_grid(label_grid, "D6", scale_factor=0.8)
        self.place_at_grid(grid_asset, "E5", scale_factor=0.5)
        
        self.play(
            j_hat.animate.scale(1.5, about_point=ORIGIN), 
            FadeIn(grid_asset),
            Write(label_grid)
        )
        self.wait(1)
