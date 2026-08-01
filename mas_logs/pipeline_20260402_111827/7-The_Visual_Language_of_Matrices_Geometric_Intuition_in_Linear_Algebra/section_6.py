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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize Scene with correct lines from prompt
        lecture_lines = [
            "Matrices move basis vectors to transform our world.",
            "Visualizing this makes linear algebra feel intuitive.",
            "Master the visual language of matrices."
        ]
        self.setup_layout("Summary & Intuition Check", lecture_lines)

        # SVG Asset for Pixel the Cat
        pixel_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/pixel.svg"

        def get_pixel():
            # Load SVG, color it, and scale for the mini-grids
            p = SVGMobject(pixel_path)
            p.set_color(BLUE_B)
            p.scale(0.25)
            return p

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Icons for vector, matrix, determinant (White #FFFFFF)
        vec_icon = Arrow(start=LEFT*0.3, end=RIGHT*0.3, color=WHITE)
        
        mat_elements = VGroup(
            Text("a", font_size=22), Text("b", font_size=22),
            Text("c", font_size=22), Text("d", font_size=22)
        ).arrange_in_grid(rows=2, cols=2, buff=0.3)
        mat_bracket_l = Text("[", font_size=42).scale(np.array([0.7, 1.3, 1])).next_to(mat_elements, LEFT, buff=0.1)
        mat_bracket_r = Text("]", font_size=42).scale(np.array([0.7, 1.3, 1])).next_to(mat_elements, RIGHT, buff=0.1)
        mat_icon = VGroup(mat_bracket_l, mat_elements, mat_bracket_r).set_color(WHITE)
        
        det_icon = Text("det(A)", font_size=22, color=WHITE)
        
        icons = VGroup(vec_icon, mat_icon, det_icon).arrange(RIGHT, buff=0.8)
        # Fix for Issue 59: Repositioned to A4-B6
        self.place_in_area(icons, 'A4', 'B6', scale_factor=0.7)
        
        self.play(FadeIn(icons))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE_B)
        self.play(FadeOut(icons))

        # Gallery of three quick grid transformations
        # 1. Rotation (Issue 57 Fix: C2-D3)
        plane1 = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], x_length=2, y_length=2, 
                            background_line_style={"stroke_opacity": 0.4})
        pixel1 = get_pixel()
        gal1 = VGroup(plane1, pixel1)
        self.place_in_area(gal1, 'C2', 'D3', scale_factor=0.6)
        label1 = Text("Rotation", font_size=16).next_to(gal1, DOWN, buff=0.1)

        # 2. Shear (Spaced out to C4-D5)
        plane2 = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], x_length=2, y_length=2, 
                            background_line_style={"stroke_opacity": 0.4})
        pixel2 = get_pixel()
        gal2 = VGroup(plane2, pixel2)
        self.place_in_area(gal2, 'C4', 'D5', scale_factor=0.6)
        label2 = Text("Shear", font_size=16).next_to(gal2, DOWN, buff=0.1)

        # 3. Flip (Spaced out to C6-D6)
        plane3 = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], x_length=2, y_length=2, 
                            background_line_style={"stroke_opacity": 0.4})
        pixel3 = get_pixel()
        gal3 = VGroup(plane3, pixel3)
        self.place_in_area(gal3, 'C6', 'D6', scale_factor=0.6)
        label3 = Text("Flip", font_size=16).next_to(gal3, DOWN, buff=0.1)

        gallery = VGroup(gal1, label1, gal2, label2, gal3, label3)
        self.play(FadeIn(gallery))

        # Perform the transformations
        rot_mat = [[0, -1], [1, 0]]
        shear_mat = [[1, 1], [0, 1]]
        flip_mat = [[-1, 0], [0, 1]]

        self.play(
            gal1.animate.apply_matrix(rot_mat),
            gal2.animate.apply_matrix(shear_mat),
            gal3.animate.apply_matrix(flip_mat),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        
        final_title = Text("The Visual Language of Matrices", font_size=32, color=WHITE)
        # Fix for Issue 58: Repositioned to E2-F5
        self.place_in_area(final_title, 'E2', 'F5', scale_factor=0.8)
        
        self.play(
            FadeOut(gallery),
            FadeIn(final_title)
        )
        self.wait(3)
