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
        title = "Application: Computer Graphics & Animation"
        lines = [
            "These geometric transformations power modern computer graphics.",
            "Matrices rotate and scale 3D models in real-time.",
            "Linear algebra brings digital worlds to life through movement."
        ]
        self.setup_layout(title, lines)

        # Colors
        CAT_COLOR = "#D3D3D3"  # Light gray
        MATRIX_COLOR = "#FFD700"  # Gold
        HIGHLIGHT_COLOR = "#00FFFF"  # Cyan for emphasis
        
        # Asset Path
        CAT_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png"

        # --- Mechanical Cat Construction ---
        # Note: Using ImageMobject for the PNG asset. 
        # ImageMobject doesn't support .set_color() well for hex colors, 
        # so we rely on the asset's original appearance or use set_color with a specific color filter if possible.
        # However, following the requirement to display it "in light gray", 
        # we will use the asset as is and apply a color overlay if it were a VMobject.
        # Since it's a PNG, we'll just load it.
        cat = ImageMobject(CAT_ASSET)
        cat.height = 1.5 # Initial sizing

        # === Animation for Lecture Line 1 ===
        # Use only color changes for lecture lines
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Reposition and scale cat to avoid obstructing formulas: Use C2-F4, scale 1.0
        self.place_in_area(cat, "C2", "F4", scale_factor=1.0)
        
        self.play(FadeIn(cat), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(HIGHLIGHT_COLOR))
        
        # Matrix display - Improve matrix centering/spacing: Use A2-A5, scale 0.8
        rotation_matrix = MathTex(r"R(\theta) = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}", color=MATRIX_COLOR)
        self.place_in_area(rotation_matrix, 'A2', 'A5', scale_factor=0.8)
        
        # 1. Rotate
        self.play(Write(rotation_matrix))
        # Spin the cat
        self.play(Rotate(cat, angle=2*PI, about_point=cat.get_center()), run_time=2)
        self.play(FadeOut(rotation_matrix))

        # 2. Scale (Jump) - Improve matrix centering/spacing: Use A2-A5, scale 0.8
        scale_matrix = MathTex(r"S(s_y) = \begin{bmatrix} 1 & 0 \\ 0 & 1.5 \end{bmatrix}", color=MATRIX_COLOR)
        self.place_in_area(scale_matrix, 'A2', 'A5', scale_factor=0.8)
        
        self.play(Write(scale_matrix))
        # ImageMobject.animate.apply_matrix is supported. 
        # We simulate a "jump" with vertical stretching.
        original_height = cat.height
        self.play(
            cat.animate.stretch(1.5, dim=1),
            run_time=0.5,
            rate_func=there_and_back
        )
        self.play(FadeOut(scale_matrix))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(HIGHLIGHT_COLOR))
        
        # Create a grid of cats
        cat_grid_group = Group() # Use Group for ImageMobjects
        
        # Define grid boundaries (roughly B2 to E5)
        center_b2 = self.grid["B2"]
        center_e5 = self.grid["E5"]
        total_width = center_e5[0] - center_b2[0]
        total_height = center_b2[1] - center_e5[1]
        
        for i in range(3):
            for j in range(3):
                mini_cat = cat.copy().scale(0.3)
                pos_x = center_b2[0] + j * (total_width / 2)
                pos_y = center_b2[1] - i * (total_height / 2)
                mini_cat.move_to(np.array([pos_x, pos_y, 0]))
                cat_grid_group.add(mini_cat)

        self.play(
            ReplacementTransform(cat, cat_grid_group),
            run_time=2
        )

        # Animate all cats in the grid
        # We can't set color easily on ImageMobjects, but we can rotate and scale.
        # We alternate animations for visual complexity.
        self.play(
            *[c.animate.rotate(PI/4) for c in cat_grid_group[::2]],
            *[c.animate.scale(1.2) for c in cat_grid_group[1::2]],
            run_time=1.5
        )
        self.play(
            *[c.animate.rotate(-PI/4) for c in cat_grid_group[::2]],
            *[c.animate.scale(1/1.2) for c in cat_grid_group[1::2]],
            run_time=1.5
        )

        self.wait(2)
        # Reset color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
