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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Synthesis", [
            "Linear transformations preserve the origin and grid lines.",
            "Matrices act as compact recipes for spatial deformation.",
            "Geometry and algebra are now perfectly linked."
        ])
        
        # Assets
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # Elements
        grid_lines = VGroup(*[Line(start=[-3, -3, 0], end=[3, -3, 0]).shift(i * UP) for i in range(7)],
                            *[Line(start=[-3, -3, 0], end=[-3, 3, 0]).shift(i * RIGHT) for i in range(7)]).set_stroke(width=1)
        grid_morphed = grid_lines.copy().apply_matrix([[1, 1], [0, 1]]).set_stroke(color="#00FF00")
        
        matrix_eq = MathTex(r"A", r"\vec{x}", "=", r"\vec{b}").scale(1.5)
        summary_label = Text("Summary", color="#FFFFFF")
        concept_label = Text("Core Concept", color="#FFFF00")
        linear_space_label = Text("Linear Space", color="#CCCCCC")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(grid_icon, "A5", scale_factor=0.5)
        self.place_at_grid(summary_label, "A4", scale_factor=0.9)
        self.play(FadeIn(grid_lines), FadeIn(grid_icon), FadeIn(summary_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.place_at_grid(concept_label, "B4", scale_factor=0.9)
        self.place_in_area(matrix_eq, "C2", "C5", scale_factor=0.8)
        self.play(Transform(grid_lines, grid_morphed), FadeIn(matrix_eq), FadeIn(concept_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#CCCCCC")
        self.place_at_grid(linear_space_label, "D4", scale_factor=0.8)
        self.place_at_grid(grid_icon.copy(), "F5", scale_factor=0.5)
        self.play(FadeIn(linear_space_label))
        self.wait(2)
