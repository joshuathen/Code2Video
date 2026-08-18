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
        self.setup_layout(
            "The Context Problem: Why Words Need Neighbors",
            [
                "Words often have multiple meanings.",
                "Context defines which meaning is correct.",
                "Attention helps models focus on relevant neighbors."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        s1_words_list = ["The", "crane", "flew", "over", "the", "construction", "site"]
        s1 = VGroup(*[Text(w, font_size=24, color="#ADD8E6") for w in s1_words_list]).arrange(RIGHT, buff=0.2)
        self.place_in_area(s1, "B1", "B6", scale_factor=0.8)
        
        self.play(Write(s1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        s2_words_list = ["The", "crane", "lifted", "the", "heavy", "beam"]
        s2 = VGroup(*[Text(w, font_size=24, color="#90EE90") for w in s2_words_list]).arrange(RIGHT, buff=0.2)
        self.place_in_area(s2, "D1", "D6", scale_factor=0.8)
        
        self.play(Write(s2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Highlight 'crane' (index 1 in both sentences)
        crane1 = s1[1]
        crane2 = s2[1]
        
        # Targets for arrows
        flew = s1[2]
        heavy_beam = VGroup(s2[4], s2[5]) # 'heavy' and 'beam'

        arrow1 = Arrow(crane1.get_bottom(), flew.get_top(), color=YELLOW, buff=0.1)
        arrow2 = Arrow(crane2.get_top(), heavy_beam.get_bottom(), color=YELLOW, buff=0.1)

        self.play(
            crane1.animate.set_color(YELLOW),
            crane2.animate.set_color(YELLOW),
            Create(arrow1),
            Create(arrow2)
        )
        self.wait(2)
        
        # Cleanup color for final state
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
