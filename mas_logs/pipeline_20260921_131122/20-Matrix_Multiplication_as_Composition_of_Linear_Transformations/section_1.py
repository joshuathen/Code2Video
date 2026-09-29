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
        lecture_lines = ["A matrix acts as a transformation function.", "Vectors change position through these functions.", "We visualize grid distortions by matrices."]
        self.setup_layout("Prerequisite Review: The Linear Transformation", lecture_lines)
        
        # Objects
        vec_v = Vector([1, 1], color="#ADD8E6")
        label_v = MathTex("v", color="#ADD8E6")
        matrix_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg", color="#FFD700")
        vec_v_prime = Vector([2, 0], color="#FF6347")
        label_v_prime = MathTex("v'", color="#FF6347")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#ADD8E6")
        self.place_at_grid(vec_v, "B2", scale_factor=0.8)
        self.place_at_grid(label_v, "B3", scale_factor=0.8)
        self.play(Create(vec_v), Write(label_v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF6347")
        self.place_at_grid(matrix_a, "D2", scale_factor=0.7)
        self.play(FadeIn(matrix_a))
        
        # Transform vector
        self.play(
            vec_v.animate.become(vec_v_prime),
            label_v.animate.move_to(self.grid["D4"]),
            FadeIn(label_v_prime.move_to(self.grid["D5"]))
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        
        # Visualize grid distortion
        grid = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], background_line_style={"stroke_opacity": 0.3})
        self.place_in_area(grid, "D4", "F6", scale_factor=0.3)
        self.play(Create(grid))
        self.play(grid.animate.apply_matrix([[1, 1], [0, 1]])) # Shear transform
        
        # Clear/remove for final state
        self.play(FadeOut(grid), FadeOut(matrix_a))
        self.wait(2)
