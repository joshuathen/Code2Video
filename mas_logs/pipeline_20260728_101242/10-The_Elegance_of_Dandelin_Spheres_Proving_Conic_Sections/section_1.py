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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initialize title and lecture lines
        title_text = "The Slice: From 3D to 2D"
        lecture_lines = [
            "- A conic section forms where a plane slices a cone.",
            "- Tilting the plane creates different geometric curves.",
            "- An ellipse is defined by a constant distance sum."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line in Green to match the plane
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Load cone SVG asset and position it
        # Issue 25: Use asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg").set_color(WHITE)
        # Issue 29: self.place_in_area(cone, 'B3', 'F5', scale_factor=1.2)
        self.place_in_area(cone, 'B3', 'F5', scale_factor=1.2)
        
        # Draw a green line representing a cutting plane
        plane_line = Line(LEFT * 2.5, RIGHT * 2.5, color="#00FF00")
        plane_line.rotate(20 * DEGREES)
        # Issue 30: self.place_in_area(plane_line, 'D3', 'E5', scale_factor=1.1)
        self.place_in_area(plane_line, 'D3', 'E5', scale_factor=1.1)
        
        self.play(Create(cone), Create(plane_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition highlight to second line (PURPLE)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#800080")
        )
        
        # Animate a purple ellipse appearing at the intersection
        ellipse = Ellipse(width=2.5, height=0.7, color="#800080")
        # Issue 31: self.place_in_area(ellipse, 'D3', 'E5', scale_factor=1.0)
        self.place_in_area(ellipse, 'D3', 'E5', scale_factor=1.0)
        # Rotate ellipse to align with the plane's tilt
        ellipse.rotate(20 * DEGREES)
        
        self.play(Create(ellipse))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlight to third line (YELLOW)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        # Highlight two yellow points F1 and F2 on the plane
        f1_dot = Dot(color="#FFFF00")
        f2_dot = Dot(color="#FFFF00")
        
        # Position foci relative to the ellipse center and rotation
        ellipse_center = ellipse.get_center()
        rot_vec = np.array([np.cos(20*DEGREES), np.sin(20*DEGREES), 0])
        
        # Foci placement
        f1_dot.move_to(ellipse_center - 0.8 * rot_vec)
        f2_dot.move_to(ellipse_center + 0.8 * rot_vec)
        
        f1_label = MathTex("F_1", color="#FFFF00", font_size=24)
        f2_label = MathTex("F_2", color="#FFFF00", font_size=24)
        
        # Labels positioned near the foci
        f1_label.next_to(f1_dot, UP, buff=0.2)
        f2_label.next_to(f2_dot, DOWN, buff=0.2)
        
        self.play(
            FadeIn(f1_dot), FadeIn(f2_dot),
            Write(f1_label), Write(f2_label)
        )
        self.wait(3)
