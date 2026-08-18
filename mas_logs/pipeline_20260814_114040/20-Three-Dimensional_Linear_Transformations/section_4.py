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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Scaling alters the basis vector lengths.",
            "Rotation changes direction, keeping orthogonality intact.",
            "Matrices represent these precise spatial changes.",
            "Model growth is a scaling matrix.",
            "Character turns use a rotation matrix."
        ]
        self.setup_layout("Visualizing Rotation and Scaling", lecture_lines)
        
        # Load asset
        char_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/character.png")
        
        # Initial state
        axes = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[-3, 3], axis_config={"include_tip": True})
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.5)
        
        # Helper for coloring
        def set_color(idx, color_hex):
            self.lecture[idx].set_color(color_hex)

        # === Animation for Lecture Line 1 ===
        set_color(0, "#FF4500")
        self.place_at_grid(char_icon, "A5", scale_factor=0.3)
        self.play(FadeIn(char_icon), axes.animate.rotate(PI/2, axis=OUT))

        # === Animation for Lecture Line 2 ===
        set_color(1, "#00FFFF")
        self.play(axes.animate.scale(2.0))

        # === Animation for Lecture Line 3 ===
        set_color(2, "#FFFFFF")
        self.play(
            axes.animate.scale(0.5).rotate(-PI/4, axis=OUT)
        )

        # === Animation for Lecture Line 4 ===
        set_color(3, "#FFD700")
        matrix_scale = MathTex(r"S = \begin{pmatrix} k & 0 \\ 0 & k \end{pmatrix}").set_color("#FFD700")
        self.place_at_grid(matrix_scale, 'B4', scale_factor=0.6)
        label_S = Text("Scaling", font_size=20).set_color("#FFD700")
        self.place_at_grid(label_S, 'B3', scale_factor=0.7)
        self.play(Write(matrix_scale), Write(label_S), FadeIn(char_icon))

        # === Animation for Lecture Line 5 ===
        set_color(4, "#FFD700")
        matrix_rot = MathTex(r"R = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}").set_color("#FFD700")
        self.place_at_grid(matrix_rot, 'C4', scale_factor=0.6)
        label_R = Text("Rotation", font_size=20).set_color("#FFD700")
        self.place_at_grid(label_R, 'C3', scale_factor=0.7)
        self.play(Write(matrix_rot), Write(label_R))
        self.wait(2)
