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
        # Data from storyboard
        lecture_lines = [
            "Legend says monks move sixty-four disks between three pillars.",
            "Only one disk moves at a time.",
            "Larger disks cannot sit on smaller ones."
        ]
        self.setup_layout("The Legend and the Rules", lecture_lines)

        # Colors from storyboard
        PILLAR_COLOR = "#808080"
        RED_COLOR = "#FF0000"
        GREEN_COLOR = "#00FF00"
        BLUE_COLOR = "#0000FF"
        GLOW_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Legend says monks move sixty-four disks between three pillars.
        self.play(self.lecture[0].animate.set_color(PILLAR_COLOR))
        
        # Create 3 pillars
        # Height 3.5 covers rows B to E (approx 3 units between centers + 0.5 padding)
        p1 = Rectangle(height=3.5, width=0.2, fill_color=PILLAR_COLOR, fill_opacity=1, stroke_width=0)
        p2 = Rectangle(height=3.5, width=0.2, fill_color=PILLAR_COLOR, fill_opacity=1, stroke_width=0)
        p3 = Rectangle(height=3.5, width=0.2, fill_color=PILLAR_COLOR, fill_opacity=1, stroke_width=0)
        
        self.place_in_area(p1, "B2", "E2")
        self.place_in_area(p2, "B4", "E4")
        self.place_in_area(p3, "B6", "E6")
        
        # Create Disks as specified
        # Heights are set to 0.6 to look good in 1.0 unit grid cells
        d_blue = RoundedRectangle(corner_radius=0.1, height=0.6, width=1.6, fill_color=BLUE_COLOR, fill_opacity=1, stroke_color=WHITE, stroke_width=1)
        d_green = RoundedRectangle(corner_radius=0.1, height=0.6, width=1.1, fill_color=GREEN_COLOR, fill_opacity=1, stroke_color=WHITE, stroke_width=1)
        d_red = RoundedRectangle(corner_radius=0.1, height=0.6, width=0.6, fill_color=RED_COLOR, fill_opacity=1, stroke_color=WHITE, stroke_width=1)
        
        # Initial stack on Pillar 1 (shifted up according to issue #33)
        self.place_at_grid(d_blue, "E2")
        self.place_at_grid(d_green, "D2")
        self.place_at_grid(d_red, "C2")
        
        self.play(
            Create(VGroup(p1, p2, p3)),
            FadeIn(VGroup(d_blue, d_green, d_red), shift=UP*0.5),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Only one disk moves at a time.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(RED_COLOR)
        )
        
        # Move smallest Red disk from P1 (C2) to P2 (E4)
        self.play(
            d_red.animate.move_to(self.grid["E4"]),
            path_arc=-PI/2,
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Larger disks cannot sit on smaller ones.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(BLUE_COLOR)
        )
        
        # Try to move largest Blue disk onto Red disk (on P2 at E4)
        # Target is D4 (cell above Red disk)
        start_pos_blue = d_blue.get_center()
        invalid_target = self.grid["D4"]
        
        self.play(
            d_blue.animate.move_to(invalid_target),
            path_arc=-PI/3,
            run_time=1.2
        )
        
        # Show red "X" to indicate illegal move
        cross = VGroup(
            Line(UP+LEFT, DOWN+RIGHT),
            Line(UP+RIGHT, DOWN+LEFT)
        ).set_color(RED_COLOR).scale(0.4).move_to(invalid_target + UP*0.5)
        
        self.play(Create(cross))
        self.play(Flash(cross, color=RED_COLOR, line_length=0.2, flash_radius=0.3))
        self.wait(0.5)
        
        # Illegal move: return Blue disk to start
        self.play(
            FadeOut(cross),
            d_blue.animate.move_to(start_pos_blue),
            path_arc=PI/3,
            run_time=1.2
        )
        self.wait(1)

        # Final setup highlight with glow
        glow = SurroundingRectangle(VGroup(p1, p2, p3, d_blue, d_green, d_red), color=GLOW_COLOR, buff=0.4, stroke_width=2)
        self.play(Create(glow))
        self.play(glow.animate.set_stroke(opacity=0.3))
        self.wait(2)
        
        # Final cleanup
        self.play(
            FadeOut(glow),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
