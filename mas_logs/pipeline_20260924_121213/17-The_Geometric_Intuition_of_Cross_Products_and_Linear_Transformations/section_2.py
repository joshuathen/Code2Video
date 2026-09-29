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

class Section2Scene(ThreeDScene, TeachingScene):
    def construct(self):
        self.setup_layout("Defining the Cross Product", [
            "Move to 3D with two distinct vectors.",
            "The cross product produces a perpendicular vector.",
            "Its magnitude is the parallelogram's surface area."
        ])
        
        # Initialize 3D space
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        axes = ThreeDAxes(x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[-3, 3, 1])
        
        # Create group for 3D elements to keep them positioned together
        three_d_group = VGroup(axes)
        
        # Assets
        u = Arrow3D(start=ORIGIN, end=np.array([1, 2, 0]), color=WHITE)
        v = Arrow3D(start=ORIGIN, end=np.array([2, -0.5, 1]), color=WHITE)
        
        # Use SVG asset as requested in Issue 17
        parallelogram = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg", color="#00FFFF")
        parallelogram_group = VGroup(parallelogram)
        
        # Labels
        vector_label = Text("Vectors u, v", font_size=24, color=WHITE)
        cross_product_label = Text("Cross Product", font_size=24, color="#FFFF00")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(u), FadeIn(v))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        cross_vec = Arrow3D(start=ORIGIN, end=np.array([0.5, -0.5, 2.5]), color="#FFFF00")
        self.play(Create(cross_vec))
        self.lecture[1].set_color("#00FFFF")
        
        # Placing labels per VideoCritic
        self.place_at_grid(vector_label, 'B4', scale_factor=0.8)
        self.add(vector_label)
        self.place_at_grid(cross_product_label, 'A5', scale_factor=0.7)
        self.add(cross_product_label)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_in_area(parallelogram_group, 'D1', 'F3', scale_factor=0.6)
        self.play(Create(parallelogram_group))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
