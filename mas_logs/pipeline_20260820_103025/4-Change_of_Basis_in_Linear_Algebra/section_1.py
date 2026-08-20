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
        lecture_lines = [
            "Space is the same; coordinate descriptions differ.",
            "Imagine a point viewed through different rulers.",
            "Standard grid lines define our default perspective.",
            "Switching basis changes the coordinate numbers.",
            "The vector stays put; labels just shift."
        ]
        self.setup_layout("The Concept of 'Perspective'", lecture_lines)
        
        # Initialize objects
        grid_a = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"color": "#3498DB"}).scale(0.4)
        grid_b = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"color": "#E74C3C"}).scale(0.4).rotate(30*DEGREES)
        vector_v = Vector([1, 1], color="#F1C40F")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        label_va = Text("v_A", font_size=20, color="#3498DB")
        label_vb = Text("v_B", font_size=20, color="#E74C3C")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        self.place_in_area(grid_a, "B3", "E5", scale_factor=0.6)
        self.place_in_area(grid_b, "B3", "E5", scale_factor=0.6)
        self.play(FadeIn(grid_a), FadeIn(grid_b))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        self.place_at_grid(vector_v, "D4", scale_factor=0.7)
        self.place_at_grid(ruler, "B1", scale_factor=0.5)
        self.play(FadeIn(vector_v), FadeIn(ruler))
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        label_va.next_to(grid_a, UP, buff=0.1)
        label_vb.next_to(grid_b, DOWN, buff=0.1)
        self.play(FadeIn(label_va), FadeIn(label_vb))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.play(Indicate(label_va), Indicate(label_vb))
        self.wait(2)
