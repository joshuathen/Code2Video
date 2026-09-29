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
        self.setup_layout("Prerequisites: The Geometry of Complex Numbers", [
            "Complex numbers form the Argand plane.",
            "Numbers act as vectors on this plane.",
            "Multiplication by i rotates by ninety degrees."
        ])
        
        # Define the axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4.0,
            y_length=4.0,
            axis_config={"color": "#444444"}
        )
        self.place_in_area(axes, 'C2', 'F5', scale_factor=0.9)
        
        # Compass Asset
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, "A6", scale_factor=0.5)

        # z1 point
        z1_pos = axes.c2p(2, 1)
        z1 = Dot(z1_pos, color="#FF00FF")
        z1_label = MathTex("z_1", color="#FF00FF")
        self.place_at_grid(z1_label, 'D5', scale_factor=0.8)
        
        # z2 point
        z2_pos = axes.c2p(-1, 2)
        z2 = Dot(z2_pos, color="#00FFFF")
        z2_label = MathTex("z_2", color="#00FFFF")
        self.place_at_grid(z2_label, 'B4', scale_factor=0.8)
        
        # Vector for z1
        vec1 = Arrow(start=axes.c2p(0, 0), end=z1_pos, color="#FFFF00", buff=0)
        
        # Polar label
        polar_label = MathTex("r e^{i\\theta}", font_size=20, color="#FFFFFF")
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(axes), FadeIn(compass))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        self.play(Create(z1), Create(z2), Write(z1_label), Write(z2_label))
        self.play(Create(vec1))
        self.place_at_grid(polar_label, 'B3', scale_factor=0.7)
        self.play(Write(polar_label))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        rotated_vec = Rotate(vec1, angle=PI/2, about_point=axes.c2p(0,0))
        rotated_z1 = Rotate(z1, angle=PI/2, about_point=axes.c2p(0,0))
        self.play(rotated_vec, rotated_z1)
        self.wait(2)
