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
        # Configuration
        title_str = "The Grand Introduction: The Most Beautiful Equation"
        lecture_lines = [
            "Euler's Identity is the world's most beautiful mathematical equation.",
            "It links five fundamental constants in one elegant formula.",
            "Each constant represents a unique branch of mathematical thought.",
            "Together, they form a bridge between algebra and geometry.",
            "Let's explore how these diverse islands of math unite."
        ]
        
        # Setup layout
        self.setup_layout(title_str, lecture_lines)
        
        # Colors for constants
        COLOR_E = "#00BFFF"     # Growth
        COLOR_I = "#FF1493"     # Rotation
        COLOR_PI = "#32CD32"    # Ratio
        COLOR_1 = "#FFD700"     # Identity
        COLOR_0 = "#FFFFFF"     # Void
        COLOR_BRIDGE = "#FFD700" # Golden Bridge

        # === Animation for Lecture Line 1 ===
        # Equation: e^{iπ} + 1 = 0
        # Building the equation with individual Text objects for precise control
        e_tex = Text("e", font_size=72)
        i_tex = Text("i", font_size=44)
        pi_tex = Text("π", font_size=44)
        plus_tex = Text("+", font_size=72)
        one_tex = Text("1", font_size=72)
        equals_tex = Text("=", font_size=72)
        zero_tex = Text("0", font_size=72)
        
        # Manual positioning to simulate MathTex layout
        exponent = VGroup(i_tex, pi_tex).arrange(RIGHT, buff=0.05)
        plus_tex.next_to(e_tex, RIGHT, buff=0.8)
        one_tex.next_to(plus_tex, RIGHT, buff=0.3)
        equals_tex.next_to(one_tex, RIGHT, buff=0.3)
        zero_tex.next_to(equals_tex, RIGHT, buff=0.3)
        exponent.next_to(e_tex.get_corner(UR), RIGHT, buff=0.05).shift(UP*0.25)
        
        equation = VGroup(e_tex, i_tex, pi_tex, plus_tex, one_tex, equals_tex, zero_tex)
        # Fix for Issue 24 and 25: use place_in_area and scale 1.5
        self.place_in_area(equation, 'C1', 'D6', scale_factor=1.5)
        
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Write(equation))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        # Highlight constants
        self.play(
            e_tex.animate.set_color(COLOR_E),
            i_tex.animate.set_color(COLOR_I),
            pi_tex.animate.set_color(COLOR_PI),
            one_tex.animate.set_color(COLOR_1),
            zero_tex.animate.set_color(COLOR_0)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Labels for constants
        labels = VGroup(
            Text("Growth", font_size=24, color=COLOR_E),
            Text("Rotation", font_size=24, color=COLOR_I),
            Text("Ratio", font_size=24, color=COLOR_PI),
            Text("Identity", font_size=24, color=COLOR_1),
            Text("Void", font_size=24, color=COLOR_0)
        )
        
        # Position labels near constants in the equation
        labels[0].next_to(e_tex, UP, buff=0.5)
        labels[1].next_to(i_tex, UP, buff=0.8)
        labels[2].next_to(pi_tex, UP, buff=0.5)
        labels[3].next_to(one_tex, DOWN, buff=0.5)
        labels[4].next_to(zero_tex, DOWN, buff=0.5)
        
        self.play(FadeIn(labels, shift=UP*0.2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_BRIDGE)
        )
        
        # Calculate bridge center (center of right side area C1-D6)
        tl_pos = self.grid["C1"]
        br_pos = self.grid["D6"]
        bridge_center = (tl_pos + br_pos) / 2
        
        # Create golden bridge circle
        bridge_circle = Circle(radius=1.8, color=COLOR_BRIDGE, stroke_width=4).move_to(bridge_center)
        bridge_glow = Circle(radius=1.8, color=COLOR_BRIDGE, stroke_width=12, stroke_opacity=0.2).move_to(bridge_center)
        
        # Points on circle for the 5 constants
        # Arranging them at equal intervals
        angles = [90, 162, 234, 306, 18] # in degrees
        target_pos = [bridge_circle.point_at_angle(a * DEGREES) for a in angles]
        
        # Transform constants and labels to circular arrangement
        # index mapping: 0:e, 1:i, 2:pi, 3:plus, 4:one, 5:equals, 6:zero
        self.play(
            FadeOut(plus_tex),
            FadeOut(equals_tex),
            e_tex.animate.move_to(target_pos[0]),
            i_tex.animate.move_to(target_pos[1]),
            pi_tex.animate.move_to(target_pos[2]),
            one_tex.animate.move_to(target_pos[3]),
            zero_tex.animate.move_to(target_pos[4]),
            # Update labels to follow their constants
            *[labels[j].animate.next_to(target_pos[j], direction=target_pos[j]-bridge_center, buff=0.3) for j in range(5)],
            Create(bridge_circle),
            FadeIn(bridge_glow),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Add islands (subtle circles around constants)
        islands = VGroup(*[
            Circle(radius=0.4, color=WHITE, stroke_opacity=0.3).move_to(pos)
            for pos in target_pos
        ])
        self.play(FadeIn(islands))
        self.wait(2)
