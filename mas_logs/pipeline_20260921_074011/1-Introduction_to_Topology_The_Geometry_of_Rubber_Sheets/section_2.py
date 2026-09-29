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
        lecture_lines = [
            "Homeomorphism describes equivalent shapes.",
            "Shapes deform without tearing or gluing.",
            "Stretching and bending preserve topology.",
            "A circle transforms into a square.",
            "These shapes are topologically identical."
        ]
        self.setup_layout("The Core Concept: Homeomorphism", lecture_lines)
        
        # Assets
        mug = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mug.svg", color="#2ECC71")
        doughnut = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/doughnut.svg", color="#2ECC71")
        square = Square(side_length=2.0, color="#2ECC71", fill_opacity=0.5)
        circle = Circle(radius=1.0, color="#2ECC71", fill_opacity=0.5)
        
        # Position initial objects
        self.place_at_grid(mug, "C4", scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(mug))
        self.lecture[0].set_color("#2ECC71")
        
        # === Animation for Lecture Line 2 ===
        self.wait(1)
        self.lecture[1].set_color("#2ECC71")
        
        # === Animation for Lecture Line 3 ===
        self.wait(1)
        self.lecture[2].set_color("#2ECC71")
        
        # === Animation for Lecture Line 4 ===
        # Morph mug (as the starting shape) into a circle, then square
        self.play(Transform(mug, circle))
        self.wait(0.5)
        self.play(Transform(mug, square))
        self.lecture[3].set_color("#2ECC71")
        
        # === Animation for Lecture Line 5 ===
        self.wait(1)
        self.play(Transform(mug, doughnut))
        self.lecture[4].set_color("#2ECC71")
        self.wait(2)
