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
        lecture_lines = [
            "Cross products create vectors perpendicular to a plane.",
            "The magnitude equals the area of the spanned parallelogram.",
            "Visualize it as a flagpole rising from the floor."
        ]
        self.setup_layout("Defining the Cross Product in 3D", lecture_lines)
        
        # 3D Setup
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2], axis_config={"include_tip": True})
        # Applying fix for Issue 24: axes position D4
        self.place_at_grid(axes, 'D4', scale_factor=0.7) 
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        u = Vector([1, 0, 0], color="#FF4500")
        v = Vector([0, 1, 0], color="#FF4500")
        w = Vector([0, 0, 1], color="#FFFF00") # Cross product result
        
        # Asset: Flagpole
        flagpole = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/flagpole.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF4500")
        self.add(axes, u, v)
        self.place_at_grid(flagpole, 'B4', scale_factor=0.3)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#1E90FF")
        # Creating a parallelogram (simplified for 3D)
        parallelogram = Polygon(
            axes.c2p(0, 0, 0), axes.c2p(1, 0, 0),
            axes.c2p(1, 1, 0), axes.c2p(0, 1, 0),
            color="#1E90FF", fill_opacity=0.3
        )
        # Applying fix for Issue 26
        self.place_in_area(parallelogram, 'E2', 'F4', scale_factor=0.5)
        self.add(parallelogram)
        
        formula_label = MathTex(r"||u \times v|| = ||u|| ||v|| \sin(\theta)", color="#FF69B4").scale(0.7)
        # Applying fix for Issue 25
        self.place_at_grid(formula_label, 'D3', scale_factor=0.9)
        self.add_fixed_in_frame_mobjects(formula_label)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.play(Create(w))
        
        normal_label = MathTex(r"\mathbf{n} = \mathbf{u} \times \mathbf{v}", color="#FFFFFF").scale(0.6)
        self.add_fixed_in_frame_mobjects(normal_label)
        normal_label.to_corner(UR)
        
        self.wait(2)
