from manim import *
import numpy as np

# Base class as specified in the prompt
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
        # Data from shared state
        title_text = "Real-World Application & Summary"
        lecture_lines = [
            "Hamming codes enable 'Self-Healing' data in critical systems.",
            "ECC RAM uses this logic to prevent system crashes.",
            "Mathematics ensures our digital world stays reliable and accurate."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Color definitions for lecture highlights
        # Matching animation colors as per requirement
        COLOR_1 = WHITE
        COLOR_2 = "#00FF00"  # Green
        COLOR_3 = WHITE
        
        # === Animation for Lecture Line 1 ===
        # Show a server rack graphic [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg] 
        # with 'ECC RAM' labeled in bright white.
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        # Asset integration (L009)
        server_rack = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg")
        server_rack.set_color(GREY_B)
        
        # Position fix (Issue 41)
        self.place_in_area(server_rack, 'B4', 'E6', scale_factor=0.8)
        
        ecc_label = Text("ECC RAM", font_size=28, color=WHITE)
        # Position label fix (Issue 41)
        self.place_at_grid(ecc_label, 'B4', scale_factor=0.7)
        
        self.play(
            FadeIn(server_rack),
            Write(ecc_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Fade in a cosmic ray symbol (a small star) [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/star.svg]
        # hitting a bit in memory and it self-correcting.
        self.play(self.lecture[1].animate.set_color(COLOR_2))
        
        # Transitions
        self.play(FadeOut(server_rack), FadeOut(ecc_label))
        
        # Bit representation - Position fix (Issue 42)
        bit_box = Square(side_length=1.4, color=WHITE, stroke_width=4)
        bit_val = Text("0", font_size=48, color=WHITE)
        bit_group = VGroup(bit_box, bit_val)
        self.place_at_grid(bit_group, 'C5', scale_factor=1.0)
        
        # Cosmic ray asset (L009)
        ray = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/star.svg")
        ray.set_color(YELLOW)
        # Position outside area initially
        self.place_at_grid(ray, 'B6', scale_factor=0.5) 
        
        self.play(FadeIn(bit_group))
        self.wait(0.5)
        
        # Ray strikes the bit
        self.play(
            ray.animate.move_to(self.grid['C5']),
            run_time=1,
            rate_func=rush_into
        )
        
        # Bit Error (Turns Red)
        error_val = Text("1", font_size=48, color=RED)
        # Position fix (Issue 42)
        self.place_at_grid(error_val, 'C5', scale_factor=1.0)
        
        self.play(
            FadeOut(ray),
            Transform(bit_val, error_val),
            bit_box.animate.set_color(RED),
            run_time=0.4
        )
        self.wait(0.8)
        
        # Self-Correction (Turns Green)
        # Use Indicate for highlighting (L004)
        self.play(Indicate(bit_group, color=COLOR_2, scale_factor=1.2))
        
        corrected_val = Text("0", font_size=48, color=COLOR_2)
        # Position fix (Issue 42)
        self.place_at_grid(corrected_val, 'C5', scale_factor=1.0)
        
        self.play(
            Transform(bit_val, corrected_val),
            bit_box.animate.set_color(COLOR_2),
            run_time=0.8
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display the text 'Self-Healing Data' in a large, bold font at the center.
        self.play(self.lecture[2].animate.set_color(COLOR_3))
        
        # Fade out previous visuals
        self.play(FadeOut(bit_group))
        
        # Large bold text as requested
        healing_text = Text("Self-Healing Data", font_size=44, color=WHITE, weight=BOLD)
        # Position fix (Issue 40)
        self.place_in_area(healing_text, 'C4', 'E6', scale_factor=1.0)
        
        self.play(Write(healing_text), run_time=2)
        self.wait(3)
