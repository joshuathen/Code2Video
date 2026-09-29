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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application: The Dimensional Shift Puzzle", [
            "Align 2D shapes to solve puzzles.",
            "Create 3D silhouettes to open locks.",
            "Dimensional shifts unlock new possibilities."
        ])

        # Assets
        cube_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        
        # Animation objects
        square = Square(color=WHITE, fill_opacity=0.5)
        cube_3d = Cube(fill_color=YELLOW, fill_opacity=0.7, stroke_width=2)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_in_area(square, 'C4', 'D5', scale_factor=0.8)
        self.play(Create(square))
        self.wait(1)
        # Using cube_asset for transition
        self.place_in_area(cube_asset, 'C4', 'D5', scale_factor=0.8)
        self.play(Transform(square, cube_asset))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.place_in_area(cube_3d, 'C4', 'D5', scale_factor=0.8)
        self.play(FadeIn(cube_3d))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        # Final object presentation using asset again
        final_cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        self.place_in_area(final_cube, 'C4', 'D5', scale_factor=1.0)
        final_cube.set_color("#FF00FF")
        self.play(Rotate(final_cube, angle=PI/2, axis=UP))
        self.wait(2)
