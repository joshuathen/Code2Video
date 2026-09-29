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
        lecture_lines = ["Meet the complex plane: real and imaginary axes.", "Think of complex numbers as vectors.", "A point like 3+4i defines a position."]
        self.setup_layout("Prerequisite: The Complex Plane", lecture_lines)
        
        # Setup axes
        axes = Axes(
            x_range=[-1, 6, 1],
            y_range=[-1, 6, 1],
            axis_config={"color": WHITE}
        )
        real_label = Text("Real", font_size=20, color=WHITE).next_to(axes.x_axis.get_end(), RIGHT)
        imag_label = Text("Imaginary", font_size=20, color=WHITE).next_to(axes.y_axis.get_end(), UP)
        plane = VGroup(axes, real_label, imag_label)
        
        # 1. Fade in the complex plane
        # === Animation for Lecture Line 1 ===
        self.place_in_area(plane, 'A3', 'F6', scale_factor=0.5)
        self.play(Write(plane))
        self.lecture[0].set_color("#FFFFFF")

        # 2. Plot a point z = x + iy on the complex plane
        # === Animation for Lecture Line 2 ===
        point_z = Dot(axes.c2p(3, 4), color="#00FF00")
        label_z = MathTex("z = 3+4i", font_size=24, color="#00FF00")
        self.place_at_grid(label_z, 'E5', scale_factor=0.7)
        
        self.play(FadeIn(point_z), Write(label_z))
        self.lecture[1].set_color("#00FF00")

        # 3. Draw a vector from the origin to z
        # === Animation for Lecture Line 3 ===
        vec = Arrow(start=axes.c2p(0, 0), end=axes.c2p(3, 4), buff=0, color="#FF0000")
        vector_group = VGroup(vec)
        self.place_in_area(vector_group, 'B3', 'E5', scale_factor=0.6)
        
        self.play(Create(vector_group))
        self.lecture[2].set_color("#FF0000")
        
        self.wait(2)
