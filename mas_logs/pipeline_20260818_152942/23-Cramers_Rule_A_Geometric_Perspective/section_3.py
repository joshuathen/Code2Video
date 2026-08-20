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
            "This creates a new, scaled parallelogram.",
            "The ratio of areas reveals the weight.",
            "Area ratio acts as a scalar.",
            "This is the heart of Cramer's Rule."
        ]
        self.setup_layout("Geometric Substitution (Area-Ratio Logic)", lecture_lines)

        # Assets
        parallelogram_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg")
        column_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/column.svg")
        vector_b_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg")
        scalar_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scalar.svg")
        
        grid_title = Text("Geometric Visualization", font_size=20)
        self.place_in_area(grid_title, 'A4', 'A6', scale_factor=0.6)
        
        main_visual_group = VGroup(parallelogram_asset, column_asset, vector_b_asset)
        self.place_in_area(main_visual_group, 'C3', 'E5', scale_factor=0.9)
        
        vector_b = vector_b_asset.copy()
        self.place_at_grid(vector_b, 'C4', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), FadeIn(parallelogram_asset))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"), FadeIn(column_asset), FadeIn(vector_b))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"), FadeIn(scalar_asset))
        
        self.wait(2)
