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
        lecture_lines = [
            "Objects cast shadows, revealing their shape.",
            "3D objects project onto 2D planes.",
            "Rotation changes the 2D shadow profile.",
            "Shadows help us infer 3D structure.",
            "This is the key to dimensional puzzles."
        ]
        self.setup_layout("Prerequisite: The Concept of Projections", lecture_lines)
        
        # Asset path
        cube_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg"
        
        # === Animation for Lecture Line 1 ===
        # Create a 3D cube using the provided SVG asset
        cube = SVGMobject(cube_asset, color=WHITE)
        self.place_at_grid(cube, 'B5', scale_factor=0.7)
        self.play(FadeIn(cube), self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        # Project a shadow of the cube onto a 2D plane
        projection_plane = Polygon(
            np.array([-1, -1, 0]), np.array([1, -1, 0]), 
            np.array([1, 1, 0]), np.array([-1, 1, 0]),
            color="#FFFF00", fill_opacity=0.3
        )
        self.place_at_grid(projection_plane, 'E3', scale_factor=0.6)
        
        shadow = Rectangle(width=1.2, height=1.2, color="#FFFF00", fill_opacity=0.8)
        self.place_at_grid(shadow, 'E5', scale_factor=0.6)
        
        self.play(FadeIn(projection_plane), FadeIn(shadow), self.lecture[1].animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 3 ===
        # Rotate the cube using the asset again
        rotated_cube = SVGMobject(cube_asset, color="#00FFFF")
        self.place_at_grid(rotated_cube, 'B5', scale_factor=0.7)
        
        self.play(
            ReplacementTransform(cube, rotated_cube),
            Rotate(rotated_cube, angle=PI/4),
            self.lecture[2].animate.set_color("#00FFFF")
        )
        # Update shadow (scale it to simulate projection change)
        self.play(shadow.animate.scale(1.2), run_time=1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        self.wait(2)
