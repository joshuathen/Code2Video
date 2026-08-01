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
        # Initial Setup
        title = "The Privacy Paradox"
        lines = [
            'Meet Pip and Leo, two users in the forest.',
            "Centralized systems rely on a single 'Forest King' server.",
            'These models track every encounter, risking a surveillance state.',
            'DP-3T replaces central tracking with private, local protection.',
            'This ensures public health without sacrificing individual privacy.'
        ]
        self.setup_layout(title, lines)
        
        # Define Colors for Lecture Lines
        line_colors = [BLUE_B, RED_B, RED_C, GREEN_B, GREEN_C]

        # === Animation for Lecture Line 1 ===
        # Pip (Penguin) and Leo (Lion) icons appear
        self.lecture[0].set_color(line_colors[0])
        
        pip = VGroup(
            Circle(radius=0.4, color=BLUE, fill_opacity=0.6),
            Text("Pip", font_size=18).shift(DOWN * 0.6)
        )
        leo = VGroup(
            Circle(radius=0.4, color=ORANGE, fill_opacity=0.6),
            Text("Leo", font_size=18).shift(DOWN * 0.6)
        )
        
        # Fixed Issue 32 & 34: Repositioned and scaled characters
        self.place_at_grid(pip, "B2", scale_factor=0.8)
        self.place_at_grid(leo, "E5", scale_factor=0.8)
        
        self.play(FadeIn(pip), FadeIn(leo))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Centralized server appears
        self.lecture[1].set_color(line_colors[1])
        
        server = VGroup(
            Square(side_length=0.8, color="#FF4444", fill_opacity=0.8),
            Text("Forest King", font_size=16, color="#FF4444").shift(UP * 0.7)
        )
        self.place_in_area(server, "C3", "D4")
        
        self.play(FadeIn(server))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Characters move and connect to server
        self.lecture[2].set_color(line_colors[2])
        
        # Move Pip and Leo closer to demonstrate interaction tracking
        self.play(
            pip.animate.move_to(self.grid["C2"]),
            leo.animate.move_to(self.grid["D5"]),
            run_time=1.5
        )
        
        # Connections to server representing surveillance
        line_pip = Line(pip.get_center(), server.get_center(), color=RED_A, stroke_width=2)
        line_leo = Line(leo.get_center(), server.get_center(), color=RED_A, stroke_width=2)
        
        self.play(Create(line_pip), Create(line_leo))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Server fades; private circles appear
        self.lecture[3].set_color(line_colors[3])
        
        # Green "Private" circles indicating local protection
        priv_pip = Circle(radius=0.6, color="#00FF00", stroke_width=4).move_to(pip.get_center())
        priv_leo = Circle(radius=0.6, color="#00FF00", stroke_width=4).move_to(leo.get_center())
        priv_text_pip = Text("Private", font_size=12, color="#00FF00").next_to(priv_pip, UP, buff=0.1)
        priv_text_leo = Text("Private", font_size=12, color="#00FF00").next_to(priv_leo, UP, buff=0.1)
        
        self.play(
            FadeOut(server),
            FadeOut(line_pip),
            FadeOut(line_leo),
            Create(priv_pip),
            Create(priv_leo),
            Write(priv_text_pip),
            Write(priv_text_leo)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # DP-3T label appears in a non-overlapping area
        self.lecture[4].set_color(line_colors[4])
        
        dp3t_label = Text("DP-3T: Privacy-First", font_size=24, color=WHITE)
        # Fixed Issue 33: Moved label to top center of the animation area
        self.place_in_area(dp3t_label, "A2", "B5", scale_factor=0.7)

        self.play(Write(dp3t_label))
        self.wait(2)
