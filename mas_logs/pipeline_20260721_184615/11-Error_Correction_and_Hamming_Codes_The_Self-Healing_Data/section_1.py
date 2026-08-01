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
        # Setup the layout with the title and lecture lines
        title_str = "The Problem: The Noisy Universe"
        lecture_lines = [
            "Radiation can flip bits during data transmission.",
            "A single bit flip changes the message meaning.",
            "Without checks, the receiver cannot detect errors.",
            "This corruption causes systems to behave unexpectedly.",
            "Hamming codes detect and fix these silent errors."
        ]
        self.setup_layout(title_str, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show a binary string "1011" (white #FFFFFF) drifting through a noisy gray background (#333333).
        # Change Line 1 color to #FFFF00. self.wait(2).
        
        # Background rect to represent noisy environment (L008: Hex colors)
        # Using A1 to F6 for the interactive area
        bg_rect = Rectangle(width=6.0, height=6.0, fill_color="#333333", fill_opacity=0.6, stroke_width=0)
        self.place_in_area(bg_rect, "A1", "F6")
        
        # Create some noise dots
        noise_dots = VGroup(*[
            Dot(radius=0.015, color="#555555").move_to([
                np.random.uniform(0.5, 5.5), 
                np.random.uniform(-2.8, 2.2), 
                0
            ])
            for _ in range(30)
        ])
        
        # Use simple Text for Math labels if needed, but MathTex is generally fine for single characters
        binary_string = VGroup(
            Text("1", color="#FFFFFF", font_size=72),
            Text("0", color="#FFFFFF", font_size=72),
            Text("1", color="#FFFFFF", font_size=72),
            Text("1", color="#FFFFFF", font_size=72)
        ).arrange(RIGHT, buff=0.4)
        
        self.place_in_area(binary_string, "B2", "D5")
        
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        self.play(FadeIn(bg_rect), Create(noise_dots))
        self.play(Write(binary_string))
        
        # Drifting effect (L031: Avoid opacity errors, use set_fill/stroke if needed)
        # We'll use a simple ValueTracker for the shift to avoid expensive always_redraw
        drift_tracker = ValueTracker(0)
        binary_string.add_updater(lambda m, dt: m.shift(LEFT * 0.1 * dt))
        
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # A yellow bolt [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bolt.svg] (#FFFF00) 
        # strikes the third bit, flipping it from '1' to '0'. 
        # Change Line 2 color to #FFFF00. self.wait(2).
        
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Issue 29: Move 'Turn Left' to A3
        left_label = Text("Turn Left", font_size=24, color="#FFFFFF")
        self.place_at_grid(left_label, "A3", scale_factor=0.8) # Corrected positioning per Issue 29
        self.play(FadeIn(left_label))
        
        # Stop drifting for the strike
        binary_string.clear_updaters()
        
        # Load bolt asset (Issue 26)
        bolt = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bolt.svg")
        bolt.set_color("#FFFF00")
        # Position bolt above the third bit
        bolt.scale(0.5).next_to(binary_string[2], UP, buff=0.5)
        
        self.play(FadeIn(bolt, shift=DOWN))
        
        # Flipping bit 1 -> 0
        new_bit = Text("0", color="#FFFFFF", font_size=72).move_to(binary_string[2])
        
        # Issue 29: Change to 'Turn Right' at A3
        right_label = Text("Turn Right", font_size=24, color="#FF0000")
        self.place_at_grid(right_label, "A3", scale_factor=0.8)
        
        self.play(
            Transform(binary_string[2], new_bit),
            ReplacementTransform(left_label, right_label),
            Flash(binary_string[2], color="#FFFF00", line_length=0.3)
        )
        self.play(FadeOut(bolt))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # The string "1011" morphs into "1001" with the '0' highlighted in red (#FF0000).
        # Change Line 3 color to #FFFF00. self.wait(2).
        
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Highlight the error bit in red
        self.play(binary_string[2].animate.set_color("#FF0000"))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # A robot graphic [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg] 
        # appears, displaying a '?' symbol above its head. 
        # Change Line 4 color to #FFFF00. self.wait(2).
        
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        
        # Load robot asset (Issue 26)
        # Issue 31: Move astro to F5 (scale 1.0)
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, "F5", scale_factor=1.0)
        
        # Question mark above head
        q_mark = Text("?", color="#FFFFFF", font_size=42).next_to(robot, UP, buff=0.1)
        
        self.play(FadeIn(robot))
        self.play(Write(q_mark))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # The red '0' flashes, then the word "HAMMING" fades in (blue #5555FF).
        # Change Line 5 color to #FFFF00. self.wait(2).
        
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        
        # Flash the red zero (L004: Indicate)
        self.play(Indicate(binary_string[2], color="#FF0000", scale_factor=1.5))
        
        # Issue 30: Move hamming_logo to A5 (scale 0.7)
        hamming_logo = Text("HAMMING", color="#5555FF", font_size=48)
        self.place_at_grid(hamming_logo, "A5", scale_factor=0.7)
        
        self.play(FadeIn(hamming_logo))
        self.wait(2)
