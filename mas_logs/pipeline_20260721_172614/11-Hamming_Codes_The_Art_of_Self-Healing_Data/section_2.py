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
        title = "Prerequisite: The Simple Parity Bit"
        lines = [
            "A parity bit helps detect single errors.",
            "We add one bit to make '1's even.",
            "If the count is odd, we know something broke."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Display '101'
        data1 = Text("1 0 1", font_size=36)
        self.place_at_grid(data1, "B2")
        
        # Highlight two 1s
        highlight1 = data1[0]
        highlight2 = data1[-1]
        
        self.play(Write(data1))
        self.wait(0.5)
        self.play(
            highlight1.animate.set_color(BLUE_C),
            highlight2.animate.set_color(BLUE_C)
        )
        
        # Add '0' parity bit
        # ISSUE 30 FIX: parity1 at B3, scale 1.0
        parity1 = Text("0", font_size=36, color="#FFFF00")
        self.place_at_grid(parity1, "B3", scale_factor=1.0)
        
        arrow1 = Arrow(start=data1.get_right(), end=parity1.get_left(), buff=0.1)
        
        self.play(Create(arrow1), FadeIn(parity1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Show '111'
        # ISSUE 31 FIX: data2 at C2
        data2 = Text("1 1 1", font_size=36)
        self.place_at_grid(data2, "C2")
        
        # Highlight three 1s
        self.play(Write(data2))
        self.wait(0.5)
        self.play(data2.animate.set_color(BLUE_C))
        
        # Add '1' parity bit
        # ISSUE 31 FIX: parity2 at C3
        parity2 = Text("1", font_size=36, color="#FFFF00")
        self.place_at_grid(parity2, "C3")
        
        arrow2 = Arrow(start=data2.get_right(), end=parity2.get_left(), buff=0.1)
        
        self.play(Create(arrow2), FadeIn(parity2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Label the extra bit 'Parity' in #FFFF00
        label1 = Text("Parity", font_size=24, color="#FFFF00")
        label2 = Text("Parity", font_size=24, color="#FFFF00")
        
        # ISSUE 30 & 31 FIXES: label1 at A4, label2 at D4, scale 0.8
        self.place_at_grid(label1, "A4", scale_factor=0.8)
        self.place_at_grid(label2, "D4", scale_factor=0.8)
        
        self.play(Write(label1), Write(label2))
        
        # Final emphasis on the odd case
        box = SurroundingRectangle(VGroup(data2, parity2), color=RED)
        self.play(Create(box))
        
        self.wait(2)
        # Clean up highlights for consistency
        self.play(self.lecture[2].animate.set_color(WHITE), Uncreate(box))
