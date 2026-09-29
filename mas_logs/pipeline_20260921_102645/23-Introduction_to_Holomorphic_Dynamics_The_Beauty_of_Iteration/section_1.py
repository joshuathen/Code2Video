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
            "Complex numbers are vectors in the plane.",
            "Holomorphic functions map shapes while preserving angles.",
            "Iteration means repeatedly applying the same function."
        ]
        self.setup_layout("Prerequisites: The Complex Plane and Functions", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show Complex Plane with x and y axes
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_numbers": False}).scale(0.5)
        self.place_at_grid(axes, 'C4', scale_factor=0.9)
        self.play(Create(axes), run_time=1)
        self.lecture[0].set_color("#00FFFF")

        # Label Real and Imaginary axes
        real_label = Text("Real", font_size=18, color="#FFD700").next_to(axes.x_axis, RIGHT)
        imag_label = Text("Imag", font_size=18, color="#FFD700").next_to(axes.y_axis, UP)
        self.add(real_label, imag_label)

        # Create a point z = a + bi (using Asset)
        z_point = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg", color="#00FFFF")
        z_pos = axes.c2p(1, 1)
        z_point.move_to(z_pos).scale(0.2)
        self.add(z_point)
        self.play(Indicate(z_point), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF6347")
        # Placeholder for "preserving angles" visualization
        shape = Square(side_length=0.5, color="#FF6347").scale(0.5)
        self.place_at_grid(shape, 'D5', scale_factor=0.6)
        self.play(Create(shape), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#90EE90")
        # Show function f(z) = z^2 + c
        func_text = MathTex("f(z) = z^2 + c", color="#FFFFFF", font_size=32)
        self.place_at_grid(func_text, 'B3', scale_factor=0.7)
        self.play(Write(func_text), run_time=1)
        
        self.wait(2)
