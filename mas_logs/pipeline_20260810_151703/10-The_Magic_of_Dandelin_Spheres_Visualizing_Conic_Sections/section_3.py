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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Ellipse has two Dandelin spheres.",
            "Each touches the plane at a focus.",
            "Points on the ellipse follow distance rules.",
            "Distances to foci sum to a constant.",
            "This defines the ellipse perfectly."
        ]
        self.setup_layout("The Geometry of Proof (Focus: Ellipse)", lecture_lines)
        
        # Ellipse and foci
        ellipse = Ellipse(width=3.0, height=2.0, color=WHITE)
        self.place_in_area(ellipse, "C2", "E6", scale_factor=0.55)
        
        # Foci as spheres (using asset)
        f1_sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=WHITE)
        f2_sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=WHITE)
        
        f1_pos = ellipse.get_center() + LEFT * 0.75
        f2_pos = ellipse.get_center() + RIGHT * 0.75
        
        f1_sphere.move_to(f1_pos).scale(0.1)
        f2_sphere.move_to(f2_pos).scale(0.1)
        self.add(ellipse, f1_sphere, f2_sphere)
        
        # Point P (using asset)
        p = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=YELLOW)
        p.scale(0.08)
        p.move_to(ellipse.point_from_proportion(0))
        
        l1 = Line(p.get_center(), f1_sphere.get_center(), color=BLUE)
        l2 = Line(p.get_center(), f2_sphere.get_center(), color=BLUE)
        
        # Text for sum
        sum_text = MathTex(r"d_1 + d_2 = \text{const}", color="#FFFF00", font_size=24)
        self.place_at_grid(sum_text, "A2", scale_factor=0.9)

        def update_l1(m):
            m.put_start_and_end_on(p.get_center(), f1_sphere.get_center())
        def update_l2(m):
            m.put_start_and_end_on(p.get_center(), f2_sphere.get_center())
        def update_p(m):
            t = self.time * 0.2
            p.move_to(ellipse.point_from_proportion(t % 1))

        l1.add_updater(update_l1)
        l2.add_updater(update_l2)
        p.add_updater(update_p)
        self.add(l1, l2, p)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFCCCC"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(f1_sphere.animate.set_color("#00FFFF"), f2_sphere.animate.set_color("#00FFFF"))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#CCFFCC"))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        self.wait(3)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#CCCCFF"))
        self.wait(2)
