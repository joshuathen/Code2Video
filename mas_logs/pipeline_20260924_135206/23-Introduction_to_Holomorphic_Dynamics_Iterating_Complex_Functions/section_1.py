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
            "Complex numbers are points in the plane.",
            "Holomorphic functions define geometric mappings.",
            "Consider the simple map f(z) equals z squared."
        ]
        self.setup_layout("Prerequisites & The Complex Plane", lecture_lines)
        
        # Setup Axes & Grid
        axes = Axes(x_length=4, y_length=4, x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": False})
        axes.get_x_axis().set_color("#00CED1")
        axes.get_y_axis().set_color("#FF4500")
        
        grid = NumberPlane(x_length=4, y_length=4, x_range=[-3, 3], y_range=[-3, 3], background_line_style={"stroke_color": "#D3D3D3", "stroke_width": 1})
        
        complex_plane_group = VGroup(grid, axes)
        
        # Place plane per issue 22
        self.place_at_grid(complex_plane_group, 'B4', scale_factor=0.9)
        
        # Define complex number z
        z = complex(1.5, 1.0)
        z_point = Dot(axes.c2p(z.real, z.imag), color=WHITE)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg] 
        # Using a fallback to represent the asset if it's empty, or just adding the label
        z_label = MathTex("z", color=WHITE)
        self.place_at_grid(z_label, 'C5', scale_factor=0.8) # Issue 23

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(complex_plane_group))
        self.lecture[0].set_color(YELLOW)
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(z_point))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # === Animation for Lecture Line 3 ===
        self.play(Write(z_label))
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
