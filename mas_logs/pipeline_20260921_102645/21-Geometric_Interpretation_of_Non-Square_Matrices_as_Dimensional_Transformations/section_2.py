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
        lecture_lines = ["A 3x2 matrix maps 2D to 3D.", "This embeds 2D space into 3D.", "Imagine a flat plane in 3D."]
        self.setup_layout("The 'Expansion' Scenario: Embedding 2D into 3D", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Using SVG asset
        plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
        plane.set_color("#90EE90")
        self.place_in_area(plane, 'B2', 'D4', scale_factor=0.6)
        self.play(FadeIn(plane))
        self.lecture[0].set_color("#90EE90")

        # === Animation for Lecture Line 2 ===
        # Place unit vectors
        i_hat = Arrow(ORIGIN, RIGHT * 0.8, color=RED, buff=0)
        j_hat = Arrow(ORIGIN, UP * 0.8, color=BLUE, buff=0)
        
        # Attach to center
        i_hat.move_to(plane.get_center())
        j_hat.move_to(plane.get_center())
        
        self.play(Create(i_hat), Create(j_hat))
        self.lecture[1].set_color("#90EE90")

        # === Animation for Lecture Line 3 ===
        # Show grid transforming/expanding effect
        grid = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], axis_config={"stroke_opacity": 0.5}).scale(0.5)
        self.place_in_area(grid, 'B2', 'D4', scale_factor=0.6)
        
        expansion_animation = VGroup(plane, i_hat, j_hat)
        
        self.play(ReplacementTransform(expansion_animation, grid))
        self.place_at_grid(grid, 'C3', scale_factor=0.75)
        
        self.lecture[2].set_color("#90EE90")
        self.wait(1)
