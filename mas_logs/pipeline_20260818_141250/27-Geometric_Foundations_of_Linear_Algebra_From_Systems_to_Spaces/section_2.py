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
        self.setup_layout("The Role of Inverse Matrices", [
            "Invertible matrices preserve the geometric dimensions.",
            "They represent a transformation that is reversible.",
            "Non-zero determinants mean space is preserved."
        ])
        
        # Load asset images
        grid_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        grid_i = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # === Animation for Lecture Line 1 ===
        # Using area C2-E3 for grid_a (from issue 42)
        self.place_in_area(grid_a, "C2", "E3", scale_factor=1.0)
        grid_a.set_color("#FFFFFF")
        self.play(FadeIn(grid_a))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        # Using area C5-E6 for grid_i (from issue 42)
        self.place_in_area(grid_i, "C5", "E6", scale_factor=1.0)
        grid_i.set_color("#FFFFFF")
        self.play(FadeIn(grid_i))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Animate grid_a to look like grid_i (identity)
        self.play(
            grid_a.animate.set_color("#FF00FF"),
            grid_i.animate.set_color("#FF00FF")
        )
        self.play(Transform(grid_a, grid_i.copy()))
        self.lecture[2].set_color("#FF00FF")
        self.wait(2)
