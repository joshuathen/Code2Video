from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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
        # 1. Setup layout
        title_text = "Prerequisite Recap: The 'Same-Dimension' Rule"
        lecture_lines = [
            "- Square matrices transform space while keeping its dimension.",
            "- A 2x2 matrix moves vectors within the 2D plane.",
            "- Basis vectors i-hat and j-hat shift but stay flat."
        ]
        self.setup_layout(title_text, lecture_lines)

        # 2. Prepare Visual Elements
        # Plane
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 2,
                "stroke_opacity": 0.3
            }
        )
        
        # Flatland Cat Illustration [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png]
        # ImageMobject is used to satisfy Issue 19
        cat = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        cat.height = 0.8
        cat.move_to(plane.get_center())
        
        # Basis Vectors and Labels
        i_hat = Vector([1, 0], color="#FF0000") # red i-hat
        j_hat = Vector([0, 1], color="#00FF00") # green j-hat
        i_label = MathTex(r"\hat{i}", color="#FF0000", font_size=24)
        j_label = MathTex(r"\hat{j}", color="#00FF00", font_size=24)
        
        # Initial relative positioning
        i_label.next_to(i_hat.get_end(), RIGHT, buff=0.1)
        j_label.next_to(j_hat.get_end(), UP, buff=0.1)
        
        # Main visual group (Plane + Cat + Vectors + Labels)
        # Using Group instead of VGroup to accommodate ImageMobject
        visual_group = Group(plane, cat, i_hat, j_hat, i_label, j_label)
        
        # Issue 22: Position visual_group in B3-F6 with scale 0.7 to avoid lecture occlusion
        self.place_in_area(visual_group, 'B3', 'F6', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # Line 1: Square matrices transform space while keeping its dimension.
        # Action: Fade in grid and the flatland cat image.
        self.play(self.lecture[0].animate.set_color(BLUE_D)) # Color matching the plane
        self.play(FadeIn(plane), FadeIn(cat), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: A 2x2 matrix moves vectors within the 2D plane.
        # Action: Display the 2x2 rotation matrix centered in the top area.
        matrix = MathTex(
            r"R = \begin{bmatrix} \cos 45^\circ & -\sin 45^\circ \\ \sin 45^\circ & \cos 45^\circ \end{bmatrix}",
            color=WHITE, font_size=32
        )
        # Issue 23: Position matrix in area A4-A5 with scale 0.8
        self.place_in_area(matrix, 'A4', 'A5', scale_factor=0.8)
        
        self.play(self.lecture[1].animate.set_color(WHITE)) # Color matching the matrix
        self.play(Write(matrix))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: Basis vectors i-hat and j-hat shift but stay flat.
        # Action: Introduce basis vectors, then rotate the whole space.
        self.play(self.lecture[2].animate.set_color(YELLOW)) # Representing combined vectors
        self.play(
            GrowArrow(i_hat),
            FadeIn(i_label),
            GrowArrow(j_hat),
            FadeIn(j_label),
            run_time=2
        )
        self.wait(1)
        
        # Transformation: Rotate the entire space (grid + cat + vectors + labels) 45 degrees CCW
        # about_point=plane.get_center() ensures rotation around the coordinate origin
        self.play(
            Rotate(visual_group, angle=45*DEGREES, about_point=plane.get_center()),
            run_time=3
        )
        self.wait(2)
