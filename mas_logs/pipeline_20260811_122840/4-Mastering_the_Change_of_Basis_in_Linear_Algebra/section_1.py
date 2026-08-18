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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Review: Coordinate Systems as Languages", 
                          ["Basis vectors form our coordinate language.", 
                           "Changing basis changes the grid lines.", 
                           "Positions are defined by relative basis units."])
        
        # Create grids
        grid_a = Axes(x_range=[-3, 3], y_range=[-3, 3], x_length=4, y_length=4, axis_config={"include_numbers": False}).set_color(WHITE)
        grid_b = Axes(x_range=[-3, 3], y_range=[-3, 3], x_length=4, y_length=4, axis_config={"include_numbers": False}).set_color(YELLOW)
        # Skewing grid_b
        grid_b.apply_matrix(np.array([[1, 0.5], [0, 1]]))
        
        cat = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        cat_label = Text("Cat", font_size=24)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.place_in_area(grid_a, 'C2', 'F5', scale_factor=0.6)
        self.play(Create(grid_a))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Transform(grid_a, grid_b))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.place_at_grid(cat, 'E4', scale_factor=0.3)
        self.place_at_grid(cat_label, 'E5', scale_factor=0.7)
        self.play(FadeIn(cat), FadeIn(cat_label))
        self.wait(2)
