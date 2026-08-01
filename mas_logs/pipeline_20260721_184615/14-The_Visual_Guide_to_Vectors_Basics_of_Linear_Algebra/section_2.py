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
        # Fetching content for Section 2
        title_text = "Defining the Vector: Magnitude and Direction"
        lecture_lines = [
            "Meet the vector: a mathematical arrow with two traits.",
            "Magnitude represents the arrow's total length or speed.",
            "Direction shows exactly where the vector is pointing."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Hexadecimal colors per L008
        COLOR_CYAN = "#00FFFF"
        COLOR_WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Show an airplane [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/airplane.svg] 
        # with a cyan #00FFFF vector arrow representing its velocity vector.
        self.play(self.lecture[0].animate.set_color(COLOR_CYAN))
        
        # Load asset
        airplane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/airplane.svg")
        airplane.set_color(COLOR_WHITE).scale(0.3)
        # Ensure airplane is oriented towards the arrow direction (initially right)
        airplane.rotate(-90 * DEGREES) 

        # Create arrow
        v_arrow = Arrow(start=LEFT, end=RIGHT, color=COLOR_CYAN, buff=0, stroke_width=6)
        
        # Positioning per Issue 30: self.place_in_area(v_arrow, 'C2', 'C5', scale_factor=1.0)
        self.place_in_area(v_arrow, "C2", "C5", scale_factor=1.0)
        
        # Attach airplane to tail
        airplane.move_to(v_arrow.get_start())
        
        # Group for easier manipulation
        v_group = VGroup(airplane, v_arrow)
        
        self.play(FadeIn(airplane), GrowArrow(v_arrow))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Animate the cyan #00FFFF arrow stretching to show an increase in magnitude 
        # while attached to the airplane. self.wait(2).
        self.play(
            self.lecture[0].animate.set_color(COLOR_WHITE),
            self.lecture[1].animate.set_color(COLOR_CYAN)
        )
        
        # Define stretched arrow (maintains start point)
        start_pt = v_arrow.get_start()
        direction_unit = (v_arrow.get_end() - start_pt) / np.linalg.norm(v_arrow.get_end() - start_pt)
        new_end_pt = start_pt + direction_unit * 3.5 # Increased length
        stretched_arrow = Arrow(start=start_pt, end=new_end_pt, color=COLOR_CYAN, buff=0, stroke_width=6)
        
        # Magnitude indicator per Issue 31: self.place_in_area(mag_group, 'D2', 'D5', scale_factor=0.8)
        brace = Brace(stretched_arrow, direction=DOWN, color=COLOR_CYAN)
        mag_text = Text("Magnitude", font_size=24, color=COLOR_CYAN)
        mag_group = VGroup(brace, mag_text).arrange(DOWN, buff=0.1)
        self.place_in_area(mag_group, "D2", "D5", scale_factor=0.8)
        
        self.play(
            Transform(v_arrow, stretched_arrow),
            FadeIn(mag_group)
        )
        self.wait(2)
        self.play(FadeOut(mag_group))

        # === Animation for Lecture Line 3 ===
        # Rotate both the cyan #00FFFF arrow and the airplane to demonstrate a change in direction. 
        # self.wait(2).
        self.play(
            self.lecture[1].animate.set_color(COLOR_WHITE),
            self.lecture[2].animate.set_color(COLOR_CYAN)
        )
        
        # Rotate both together about the tail
        # Note: v_arrow is now the result of Transform, but we use the current mobject.
        rot_angle = 45 * DEGREES
        self.play(
            Rotate(v_group, angle=rot_angle, about_point=v_arrow.get_start())
        )
        
        # Direction indicator per Issue 32: self.place_in_area(dir_group, 'D3', 'E5', scale_factor=0.8)
        # Create an arc from the original orientation to the new one
        arc_dir = Arc(
            radius=1.2, 
            start_angle=0, 
            angle=rot_angle, 
            arc_center=v_arrow.get_start(), 
            color=COLOR_CYAN
        )
        dir_text = Text("Direction", font_size=24, color=COLOR_CYAN)
        dir_group = VGroup(arc_dir, dir_text).arrange(RIGHT, buff=0.2)
        self.place_in_area(dir_group, "D3", "E5", scale_factor=0.8)
        
        self.play(Create(arc_dir), Write(dir_text))
        self.wait(2)
        self.play(FadeOut(arc_dir), FadeOut(dir_text))

        # Final cleanup
        self.play(self.lecture[2].animate.set_color(COLOR_WHITE))
        self.wait(2)
