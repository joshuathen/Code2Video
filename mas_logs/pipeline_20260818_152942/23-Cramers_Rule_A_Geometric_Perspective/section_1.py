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
        self.setup_layout("Prerequisite Review: Determinants as Area", [
            "Determinants represent the signed area of a parallelogram.", 
            "Column vectors form the sides of this shape.", 
            "The area changes under linear transformations."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Create a unit square
        square = Square(side_length=1, color=WHITE)
        self.place_in_area(square, 'D4', 'F6', scale_factor=0.8)
        self.play(Create(square))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Represent column vectors
        v1 = Vector([1, 0], color=BLUE)
        v2 = Vector([0, 1], color=RED)
        v1.shift(square.get_corner(DL))
        v2.shift(square.get_corner(DL))
        self.play(Create(v1), Create(v2))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Transform the square to parallelogram
        # Using SVG asset for parallelogram
        parallelogram_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg")
        self.place_in_area(parallelogram_img, 'A4', 'C6', scale_factor=0.9)
        self.play(ReplacementTransform(square, parallelogram_img))
        self.lecture[2].set_color("#FF00FF")
        self.wait(1)
