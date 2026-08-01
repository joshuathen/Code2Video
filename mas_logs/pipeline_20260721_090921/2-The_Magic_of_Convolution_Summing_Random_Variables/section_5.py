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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing the Outcome: The 'Smoothing' Effect", [
            "Convolution acts like a mathematical blender for distributions.",
            "Adding two uniform distributions creates a triangular-shaped result.",
            "Adding triangular distributions starts to look like a bell curve.",
            "Repeated convolution eventually leads to a Normal distribution.",
            "This is the foundation of the Central Limit Theorem."
        ])

        # Colors
        COLOR_UNIFORM = "#FFA500"
        COLOR_TRIANGLE = "#FFD700"
        COLOR_NORMAL = "#00FA9A"

        # Assets
        BLENDER_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/blender.svg"

        # === Animation for Lecture Line 1 ===
        # Convolution acts like a mathematical blender for distributions.
        self.play(self.lecture[0].animate.set_color(COLOR_UNIFORM))
        
        square1 = Square(side_length=0.8, fill_opacity=0.6, color=COLOR_UNIFORM, fill_color=COLOR_UNIFORM)
        square2 = Square(side_length=0.8, fill_opacity=0.6, color=COLOR_UNIFORM, fill_color=COLOR_UNIFORM)
        
        self.place_at_grid(square1, 'B2')
        self.place_at_grid(square2, 'B5')

        blender_icon = SVGMobject(BLENDER_PATH).scale(0.5)
        self.place_in_area(blender_icon, 'B3', 'B4')
        blender_icon.set_color(WHITE)
        
        self.play(Create(square1), Create(square2), FadeIn(blender_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Adding two uniform distributions creates a triangular-shaped result.
        self.play(self.lecture[1].animate.set_color(COLOR_TRIANGLE))
        
        triangle = Polygon(
            [-1.5, -1.0, 0], [0, 1.0, 0], [1.5, -1.0, 0],
            fill_opacity=0.6, color=COLOR_TRIANGLE, fill_color=COLOR_TRIANGLE
        )
        self.place_in_area(triangle, 'C2', 'E5')
        
        self.play(
            ReplacementTransform(VGroup(square1, square2, blender_icon), triangle)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Adding triangular distributions starts to look like a bell curve.
        self.play(self.lecture[2].animate.set_color(COLOR_TRIANGLE))
        
        # Show "smoothing" transform
        smooth_triangle = Polygon(
            [-1.5, -1.0, 0], [-0.5, 0.4, 0], [0, 1.0, 0], [0.5, 0.4, 0], [1.5, -1.0, 0],
            fill_opacity=0.6, color=COLOR_TRIANGLE, fill_color=COLOR_TRIANGLE
        )
        self.place_in_area(smooth_triangle, 'C2', 'E5')
        
        self.play(Transform(triangle, smooth_triangle))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Repeated convolution eventually leads to a Normal distribution.
        self.play(self.lecture[3].animate.set_color(COLOR_NORMAL))
        
        # Approximate bell curve
        x_vals = np.linspace(-1.5, 1.5, 50)
        y_vals = 2.0 * np.exp(- (2.5 * x_vals**2)) - 1.0
        points = [np.array([x, y, 0]) for x, y in zip(x_vals, y_vals)]
        points.append(np.array([1.5, -1.0, 0]))
        points.append(np.array([-1.5, -1.0, 0]))
        
        normal_shape = Polygon(*points, fill_opacity=0.6, color=COLOR_NORMAL, fill_color=COLOR_NORMAL)
        self.place_in_area(normal_shape, 'C2', 'E5')
        
        self.play(
            ReplacementTransform(triangle, normal_shape)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This is the foundation of the Central Limit Theorem.
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        clt_label = Text("Central Limit Theorem", font_size=24, color=WHITE)
        self.place_in_area(clt_label, 'F2', 'F5')
        
        self.play(Write(clt_label))
        self.play(Indicate(normal_shape), Indicate(clt_label))
        self.wait(2)
