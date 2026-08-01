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
        title_text = "The Problem: The Noisy Channel"
        lecture_lines = [
            "Digital data travels through noisy channels.",
            "Noise can flip a bit from zero to one.",
            "This corruption can change the message's entire meaning."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors
        WHITE_COLOR = "#FFFFFF"
        YELLOW_COLOR = "#FFFF00"
        RED_COLOR = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Digital data travels through noisy channels.
        self.play(self.lecture[0].animate.set_color(WHITE_COLOR))

        # Digital Scroll (Rectangle)
        scroll = Rectangle(width=2.5, height=1.2, color=WHITE_COLOR, fill_opacity=0.2, fill_color=WHITE_COLOR)
        bit_objects = VGroup(*[Text(b, font_size=36, color=WHITE_COLOR) for b in ["1", "0", "1", "1"]]).arrange(RIGHT, buff=0.4)
        scroll_group = VGroup(scroll, bit_objects)
        # Issue 25: Fixed proximity by starting at C3 instead of C2
        self.place_at_grid(scroll_group, "C3", scale_factor=0.9)

        # Receiver Icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/receiver.svg]
        # Issue 21: Integrated SVG asset
        receiver_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/receiver.svg")
        receiver_label = Text("Receiver", font_size=16).next_to(receiver_svg, DOWN, buff=0.2)
        receiver = VGroup(receiver_svg, receiver_label)
        self.place_at_grid(receiver, "C6", scale_factor=0.6)

        self.play(FadeIn(scroll_group), FadeIn(receiver))
        # Moving the scroll to indicate travel towards receiver
        self.play(scroll_group.animate.move_to(self.grid["C4"]), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Noise can flip a bit from zero to one.
        self.play(self.lecture[1].animate.set_color(YELLOW_COLOR))

        # Lightning Bolt (Yellow)
        # Coordinates adjusted based on scroll position at C4
        p1 = self.grid["A4"] + UP * 0.5
        p2 = self.grid["B5"] + LEFT * 0.3
        p3 = self.grid["B4"] + RIGHT * 0.3
        p4 = bit_objects[1].get_top()
        lightning = VMobject(color=YELLOW_COLOR, stroke_width=5).set_points_as_corners([p1, p2, p3, p4])
        
        self.play(Create(lightning), run_time=0.4)
        self.play(Flash(bit_objects[1], color=YELLOW_COLOR))
        
        # Flip bit 0 -> 1 in Red
        flipped_bit = Text("1", font_size=36, color=RED_COLOR).move_to(bit_objects[1])
        old_bit = bit_objects[1]
        
        self.play(
            FadeOut(old_bit),
            FadeIn(flipped_bit),
            FadeOut(lightning),
            run_time=0.5
        )
        # Update scroll group to include the new bit for future movement
        bit_objects.remove(old_bit)
        bit_objects.add(flipped_bit)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This corruption can change the message's entire meaning.
        self.play(self.lecture[2].animate.set_color(RED_COLOR))

        # Move to receiver
        self.play(scroll_group.animate.move_to(self.grid["C6"]), run_time=1.5)
        
        # Error Label
        error_label = Text("ERROR", color=RED_COLOR, font_size=32, weight=BOLD)
        # Issue 24: Moved to E6 and scaled to avoid overlap with receiver label
        self.place_at_grid(error_label, "E6", scale_factor=0.8)
        
        self.play(Write(error_label))
        self.play(Indicate(error_label, color=RED_COLOR))
        
        self.wait(2)
