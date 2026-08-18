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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Concept of Transformation", [
            "Linear transformations map vectors while preserving structure.",
            "Grid lines remain straight and parallel after transformation.",
            "The origin must remain fixed during these transformations.",
            "Imagine the grid stretching like a rubber sheet.",
            "This geometric view is fundamental to linear algebra."
        ])
        
        # Grid visual
        grid = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], background_line_style={"stroke_opacity": 0.5})
        grid_label = Text("Grid Visual", font_size=20)
        
        # Initial placement using area constraint
        grid_group = VGroup(grid)
        self.place_in_area(grid_group, "D2", "F6", scale_factor=0.5)
        self.place_at_grid(grid_label, "D1", scale_factor=0.8)
        self.add(grid_group, grid_label)
        
        # Asset
        sheet_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sheet.svg")
        
        # === Animation for Lecture Line 1 ===
        # Draw a grid of lines in #FFFFFF using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sheet.svg].
        self.play(FadeIn(grid_group), FadeIn(sheet_asset.scale(0.5).move_to(self.grid["B3"])))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Animate grid transformation maintaining straight lines in #FF0000.
        self.lecture[1].set_color("#FF0000")
        grid_transformed = grid.copy().apply_matrix([[1.5, 0.5], [0, 1]])
        self.play(Transform(grid, grid_transformed), run_time=2)

        # === Animation for Lecture Line 3 ===
        # Highlight fixed origin at (0,0) in #00FFFF.
        self.lecture[2].set_color("#00FFFF")
        origin_dot = Dot(color="#00FFFF").move_to(grid.c2p(0, 0))
        self.play(Create(origin_dot))

        # === Animation for Lecture Line 4 ===
        # Animate grid expansion and shear like rubber sheet in #FFFF00.
        self.lecture[3].set_color("#FFFF00")
        self.play(grid.animate.apply_matrix([[1, 0.5], [0, 1.2]]), run_time=2)

        # === Animation for Lecture Line 5 ===
        # Flash grid lines to emphasize structure preservation in #FFFFFF using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sheet.svg].
        self.lecture[4].set_color("#FFFFFF")
        self.play(Flash(grid), run_time=1)
        self.wait(1)
