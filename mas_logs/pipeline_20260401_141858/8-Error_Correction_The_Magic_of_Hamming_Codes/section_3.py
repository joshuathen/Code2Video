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
            "Richard Hamming proposed using multiple overlapping parity bits.",
            "Check bits occupy positions that are powers of two.",
            "Other positions hold the actual data bits being sent.",
            "Each check bit monitors a specific subset of data.",
            "This overlapping structure allows us to pinpoint error locations."
        ]
        self.setup_layout("The Hamming Strategy: Overlapping Zones", lecture_lines)
        
        blue_color = "#00BFFF"
        gold_color = "#FFD700"

        # === Animation for Lecture Line 1 ===
        # Richard Hamming proposed using multiple overlapping parity bits.
        self.play(self.lecture[0].animate.set_color(blue_color))
        
        slots = VGroup(*[
            VGroup(
                Square(side_length=0.7, stroke_color=WHITE),
                Text(str(i+1), font_size=20, color=WHITE).shift(UP * 0.6)
            ) for i in range(7)
        ]).arrange(RIGHT, buff=0.15)
        
        # Fixed positioning per issue #33 and #34
        self.place_in_area(slots, 'D1', 'D6', scale_factor=0.8)
        
        self.play(Create(slots))
        
        guardians_indices = [0, 1, 3] # positions 1, 2, 4 (0-indexed)
        guardian_labels = VGroup()
        for idx in guardians_indices:
            label = Text("G", font_size=24, color=blue_color).move_to(slots[idx][0].get_center())
            guardian_labels.add(label)
            self.play(
                slots[idx][0].animate.set_fill(blue_color, opacity=0.3).set_stroke(blue_color),
                Write(label),
                run_time=0.4
            )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Check bits occupy positions that are powers of two.
        self.play(self.lecture[1].animate.set_color(gold_color))
        
        messengers_indices = [2, 4, 5, 6] # positions 3, 5, 6, 7
        messenger_labels = VGroup()
        for idx in messengers_indices:
            label = Text("M", font_size=24, color=gold_color).move_to(slots[idx][0].get_center())
            messenger_labels.add(label)
            self.play(
                slots[idx][0].animate.set_fill(gold_color, opacity=0.3).set_stroke(gold_color),
                Write(label),
                run_time=0.4
            )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Other positions hold the actual data bits being sent.
        # Animate a highlight from Guardian 1 to Messengers 3, 5, and 7.
        self.play(self.lecture[2].animate.set_color(blue_color))
        
        p1_indices = [0, 2, 4, 6] # 1, 3, 5, 7
        highlight1 = VGroup(*[
            SurroundingRectangle(slots[i][0], color=blue_color, buff=0.1)
            for i in p1_indices
        ])
        
        self.play(Create(highlight1))
        self.play(Indicate(slots[0], color=blue_color))
        self.play(LaggedStart(*[Indicate(slots[i], color=gold_color) for i in [2, 4, 6]], lag_ratio=0.2))
        self.wait(1)
        self.play(FadeOut(highlight1))

        # === Animation for Lecture Line 4 ===
        # Each check bit monitors a specific subset of data.
        # Animate a highlight from Guardian 2 to Messengers 3, 6, and 7.
        self.play(self.lecture[3].animate.set_color(blue_color))
        
        p2_indices = [1, 2, 5, 6] # 2, 3, 6, 7
        highlight2 = VGroup(*[
            SurroundingRectangle(slots[i][0], color=blue_color, buff=0.1)
            for i in p2_indices
        ])
        
        self.play(Create(highlight2))
        self.play(Indicate(slots[1], color=blue_color))
        self.play(LaggedStart(*[Indicate(slots[i], color=gold_color) for i in [2, 5, 6]], lag_ratio=0.2))
        self.wait(1)
        self.play(FadeOut(highlight2))

        # === Animation for Lecture Line 5 ===
        # This overlapping structure allows us to pinpoint error locations.
        # Animate a highlight from Guardian 4 to Messengers 5, 6, and 7.
        self.play(self.lecture[4].animate.set_color(blue_color))
        
        p4_indices = [3, 4, 5, 6] # 4, 5, 6, 7
        highlight4 = VGroup(*[
            SurroundingRectangle(slots[i][0], color=blue_color, buff=0.1)
            for i in p4_indices
        ])
        
        self.play(Create(highlight4))
        self.play(Indicate(slots[3], color=blue_color))
        self.play(LaggedStart(*[Indicate(slots[i], color=gold_color) for i in [4, 5, 6]], lag_ratio=0.2))
        self.wait(2)
        self.play(FadeOut(highlight4))
