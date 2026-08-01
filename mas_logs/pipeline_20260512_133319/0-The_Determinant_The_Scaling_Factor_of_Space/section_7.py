from manim import *
import numpy as np

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

class Section7Scene(TeachingScene):
    def construct(self):
        # Updated lecture lines per prompt
        lines = [
            "Determinants scale area in 2D and volume in 3D.",
            "They describe how matrices reshape the space around us.",
            "Use the determinant to master the geometry of transformation."
        ]
        self.setup_layout("Summary and 3D Intuition", lines)

        # === Animation for Lecture Line 1 ===
        # Show '2D: Area Scale' vs '3D: Volume Scale' text
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        text_2d = Text("2D: Area Scale", font_size=20, color=WHITE)
        text_3d = Text("3D: Volume Scale", font_size=20, color=WHITE)
        
        self.place_at_grid(text_2d, "B2")
        self.place_at_grid(text_3d, "B5")
        
        # Simple 2D square to show area scaling
        square = Square(side_length=1.0, color=BLUE, fill_opacity=0.3)
        rect = Rectangle(width=1.5, height=0.7, color=BLUE, fill_opacity=0.3)
        self.place_at_grid(square, "C2")
        self.place_at_grid(rect, "C2") # Ready for transformation
        
        self.play(FadeIn(text_2d), FadeIn(text_3d), Create(square))
        self.play(Transform(square, rect))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Use Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/cube.svg
        self.play(self.lecture[1].animate.set_color(TEAL))
        
        # Load and place cube asset
        cube_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cube.svg")
        cube_asset.set_color(GREEN)
        # Positioned at C4-E6 to align with row C and text_3d label
        self.place_in_area(cube_asset, "C4", "E6", scale_factor=1.5)
        
        # Create a parallelepiped state by shearing the cube asset using apply_matrix
        para_asset = cube_asset.copy().apply_matrix([[1, 0.4, 0], [0.2, 1, 0], [0, 0, 1]])
        
        self.play(DrawBorderThenFill(cube_asset))
        self.play(Transform(cube_asset, para_asset))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final summary text
        self.play(self.lecture[2].animate.set_color(PURPLE))
        
        summary_text = Text(
            "Determinant:\nThe Scaling Factor of Space", 
            font_size=28, 
            color=WHITE, 
            t2c={"Determinant": YELLOW}
        )
        # Positioned at B2-E6 to avoid lecture area
        self.place_in_area(summary_text, "B2", "E6")
        
        # Clear area then show final message
        self.play(
            FadeOut(text_2d), 
            FadeOut(text_3d), 
            FadeOut(square), 
            FadeOut(cube_asset)
        )
        self.play(Write(summary_text))
        self.wait(3)
