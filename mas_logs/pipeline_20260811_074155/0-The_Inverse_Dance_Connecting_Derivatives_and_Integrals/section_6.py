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
        # Data from storyboard and outline
        title = "Summary and Synthesis"
        lecture_lines = [
            "Derivatives and integrals form a perfect mathematical loop.",
            "One breaks things down; the other builds them up.",
            "Slope and area are fundamentally linked across all functions.",
            "From Robby's walk to leaking tanks, the connection holds.",
            "This inverse dance is the heartbeat of modern calculus."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Define Colors
        COLOR_POS = "#1E90FF"  # blue
        COLOR_VEL = "#FFFF00"  # yellow
        COLOR_DIFF = "#FF0000" # red
        COLOR_INT = "#00FF00"  # green
        
        # === Animation for Lecture Line 1 ===
        # Derivatives and integrals form a perfect mathematical loop.
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # One breaks things down; the other builds them up.
        # Display tank [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg] 
        # with 'Position' (blue #1E90FF) and 'Velocity' (yellow #FFFF00) text.
        
        pos_text = Text("Position", color=COLOR_POS)
        vel_text = Text("Velocity", color=COLOR_VEL)
        
        # Resolve Issues 34 & 35: Updated positioning
        self.place_at_grid(pos_text, 'C3', scale_factor=0.6)
        self.place_at_grid(vel_text, 'C6', scale_factor=0.6)
        
        # Resolve Issue 22: Integrate SVG asset
        # Resolve Issue 36: Updated positioning for the tank (previously 'robby')
        tank = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg")
        tank.set_color(WHITE)
        self.place_in_area(tank, 'D4', 'E5', scale_factor=1.2)

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            Write(pos_text),
            Write(vel_text),
            FadeIn(tank)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Slope and area are fundamentally linked across all functions.
        # Animate a red arrow (#FF0000) labeled 'Differentiate' moving right.
        
        # Adjusted arrow start/end to match new text positions
        diff_arrow = CurvedArrow(
            self.grid['B3'], self.grid['B6'], 
            angle=-PI/3, color=COLOR_DIFF
        )
        diff_label = Text("Differentiate", color=COLOR_DIFF, font_size=20)
        diff_label.next_to(diff_arrow, UP, buff=0.1)

        self.play(
            self.lecture[2].animate.set_color(COLOR_DIFF),
            Create(diff_arrow),
            Write(diff_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # From Robby's walk to leaking tanks, the connection holds.
        # Animate a green arrow (#00FF00) labeled 'Integrate' moving left.
        
        # Adjusted arrow start/end to clear the tank at D4-E5
        int_arrow = CurvedArrow(
            self.grid['F6'], self.grid['F3'], 
            angle=-PI/3, color=COLOR_INT
        )
        int_label = Text("Integrate", color=COLOR_INT, font_size=20)
        int_label.next_to(int_arrow, DOWN, buff=0.1)

        self.play(
            self.lecture[3].animate.set_color(COLOR_INT),
            Create(int_arrow),
            Write(int_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # This inverse dance is the heartbeat of modern calculus.
        # Heartbeat effect on the central tank
        self.play(
            self.lecture[4].animate.set_color(WHITE),
            tank.animate.scale(1.2),
            run_time=0.5
        )
        self.play(tank.animate.scale(1/1.2), run_time=0.5)
        self.play(tank.animate.scale(1.2), run_time=0.5)
        self.play(tank.animate.scale(1/1.2), run_time=0.5)
        self.wait(3)
