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
        self.setup_layout("The Tesseract Puzzle: Unfolding 4D", [
            "A cube unfolds into a 2D 'net' of squares.",
            "A 4D Tesseract unfolds into eight connected 3D cubes.",
            "This helps us visualize higher-dimensional spatial relationships.",
            "We map 4D structures into our 3D space.",
            "Complexity is simplified through dimensional unfolding."
        ])
        
        # === Animation for Lecture Line 1 ===
        # A cube unfolds into a 2D 'net' of squares.
        self.lecture_texts[0].set_color("#FFFFFF")
        cube_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        self.place_at_grid(cube_asset, "B4", scale_factor=0.6)
        self.play(FadeIn(cube_asset))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A 4D Tesseract unfolds into eight connected 3D cubes.
        self.lecture_texts[1].set_color("#00FF00")
        tesseract_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tesseract.svg")
        self.place_at_grid(tesseract_asset, "A5", scale_factor=0.6)
        self.play(FadeIn(tesseract_asset))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This helps us visualize higher-dimensional spatial relationships.
        self.lecture_texts[2].set_color("#FF00FF")
        unfolded_group = VGroup(*[Cube(side_length=0.5, stroke_width=1, color=PURPLE) for _ in range(8)])
        unfolded_group.arrange_in_grid(2, 4, buff=0.1)
        self.place_in_area(unfolded_group, "C4", "F6", scale_factor=0.5)
        self.play(FadeIn(unfolded_group))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # We map 4D structures into our 3D space.
        self.lecture_texts[3].set_color("#888888")
        grid = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], background_line_style={"stroke_opacity": 0.3}).scale(0.5)
        self.place_at_grid(grid, "D3")
        self.play(Create(grid))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Complexity is simplified through dimensional unfolding.
        self.lecture_texts[4].set_color("#FFFFFF")
        final_tesseract = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tesseract.svg")
        self.place_at_grid(final_tesseract, "C2", scale_factor=0.7)
        self.play(FadeIn(final_tesseract), Indicate(unfolded_group))
        self.wait(2)
