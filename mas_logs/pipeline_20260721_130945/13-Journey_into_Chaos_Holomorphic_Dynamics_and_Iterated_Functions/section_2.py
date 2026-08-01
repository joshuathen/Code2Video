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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "Prerequisite: The Complex Plane as a Playground"
        lecture_lines = [
            "We treat complex numbers as points on a plane.",
            "Squaring a point stretches and rotates its position.",
            "This geometry drives the motion of our dynamical system."
        ]
        
        # Colors from storyboard
        grid_color = "#FFFFFF"
        axes_label_color = "#FFD700"
        dot_color = "#FF69B4"
        
        # Initialize layout
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Draw a white coordinate grid (#FFFFFF), label the origin, and incorporate the playground icon
        plane = ComplexPlane(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            background_line_style={
                "stroke_color": grid_color,
                "stroke_width": 1,
                "stroke_opacity": 0.5
            },
            axis_config={"stroke_color": grid_color}
        )
        self.place_in_area(plane, "B2", "E5", scale_factor=0.7)
        
        origin_label = Text("0", font_size=18, color=grid_color)
        origin_label.next_to(plane.n2p(0), DL, buff=0.1)

        # Incorporate the playground icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/playground.svg]
        playground_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/playground.svg")
        self.place_at_grid(playground_icon, "A6", scale_factor=0.5)

        self.play(
            self.lecture[0].animate.set_color(grid_color),
            Create(plane),
            Write(origin_label),
            FadeIn(playground_icon),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Label the x-axis 'Real' (#FFD700) and y-axis 'Imaginary' (#FFD700).
        real_label = Text("Real", font_size=20, color=axes_label_color)
        imag_label = Text("Imaginary", font_size=20, color=axes_label_color)
        
        # Applying VideoCritic fixes:
        # Issue 25: Reposition real_label to D6
        self.place_at_grid(real_label, 'D6', scale_factor=0.5)
        # Issue 26: Reposition imag_label to A4
        self.place_at_grid(imag_label, 'A4', scale_factor=0.5)

        self.play(
            self.lecture[1].animate.set_color(axes_label_color),
            Write(real_label),
            Write(imag_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Plot a single bright dot at (1, 1) and label it z_0 = 1 + i in #FF69B4.
        z0_val = 1 + 1j
        z0_point = plane.n2p(z0_val)
        dot = Dot(z0_point, color=dot_color, radius=0.08)
        
        # Fallback to Text if MathTex fails (L022)
        try:
            z0_label = MathTex("z_0 = 1 + i", font_size=24, color=dot_color)
        except:
            z0_label = Text("z0 = 1 + i", font_size=20, color=dot_color)
            
        # Applying VideoCritic fix:
        # Issue 27: Reposition z0_label to area B5-C6
        self.place_in_area(z0_label, 'B5', 'C6', scale_factor=0.6)

        self.play(
            self.lecture[2].animate.set_color(dot_color),
            FadeIn(dot, scale=0.5),
            Write(z0_label),
            run_time=1.5
        )
        self.wait(3)
