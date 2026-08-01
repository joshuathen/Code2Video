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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initialize Scene with specified Stage-3 lecture lines
        lecture_lines = [
            'Parity bits are placed at indices of powers of two.',
            'The remaining slots are filled with the message data.',
            'Data positions are sums of their controlling parity indices.'
        ]
        self.setup_layout("Strategic Placement (Powers of 2)", lecture_lines)

        # Pre-create bit slots and labels
        # Positions: 1, 2, 3, 4, 5, 6, 7
        indices = ["1", "2", "3", "4", "5", "6", "7"]
        grid_pos = ["B1", "B2", "B3", "B4", "C1", "C2", "C3"]
        
        bit_boxes = {}
        bit_labels = {}
        slots = VGroup()
        
        for idx, g_pos in zip(indices, grid_pos):
            box = Square(side_length=0.8, color=GRAY, stroke_width=2)
            self.place_at_grid(box, g_pos)
            
            label = Text(idx, font_size=20, color=WHITE)
            label.next_to(box, UP, buff=0.1)
            
            bit_boxes[idx] = box
            bit_labels[idx] = label
            slots.add(box, label)

        # === Animation for Lecture Line 1 ===
        # Parity bits are placed at indices of powers of two.
        self.play(self.lecture[0].animate.set_color(GOLD))
        self.play(FadeIn(slots))
        
        # Highlight powers of 2 (1, 2, 4) in Gold
        parity_indices = ["1", "2", "4"]
        p_label_objs = VGroup()
        for idx in parity_indices:
            p_text = Text(f"P{idx}", font_size=24, color=GOLD)
            self.place_at_grid(p_text, grid_pos[indices.index(idx)])
            p_label_objs.add(p_text)
            self.play(
                bit_boxes[idx].animate.set_color(GOLD).set_stroke(width=4),
                FadeIn(p_text),
                run_time=0.4
            )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The remaining slots are filled with the message data.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE_B)
        )
        
        data_indices = ["3", "5", "6", "7"]
        d_label_objs = VGroup()
        for idx in data_indices:
            d_text = Text(f"D{idx}", font_size=24, color=WHITE)
            self.place_at_grid(d_text, grid_pos[indices.index(idx)])
            d_label_objs.add(d_text)
            self.play(FadeIn(d_text), run_time=0.3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Data positions are sums of their controlling parity indices.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Highlight logic: P1 checks 1, 3, 5, 7
        highlight_p1 = VGroup(*[bit_boxes[i] for i in ["1", "3", "5", "7"]])
        rect_p1 = SurroundingRectangle(highlight_p1, color=GOLD, buff=0.1)
        p1_info = Text("P1: Checks 1, 3, 5, 7", font_size=20, color=GOLD)
        # Resolved Issue 37: Place at F6, scale 0.8
        self.place_at_grid(p1_info, 'F6', scale_factor=0.8)
        
        self.play(Create(rect_p1), FadeIn(p1_info))
        self.wait(1)
        self.play(FadeOut(rect_p1), FadeOut(p1_info))

        # Highlight logic: P2 checks 2, 3, 6, 7
        highlight_p2 = VGroup(*[bit_boxes[i] for i in ["2", "3", "6", "7"]])
        rect_p2 = SurroundingRectangle(highlight_p2, color=GOLD, buff=0.1)
        p2_info = Text("P2: Checks 2, 3, 6, 7", font_size=20, color=GOLD)
        # Resolved Issue 37: Place at F6, scale 0.8
        self.place_at_grid(p2_info, 'F6', scale_factor=0.8)

        self.play(Create(rect_p2), FadeIn(p2_info))
        self.wait(1)
        self.play(FadeOut(rect_p2), FadeOut(p2_info))

        # Binary address logic for bit 3
        # Resolved Issue 38: Place at D6, scale 0.7
        bin_addr = Text("3 = 011 (binary)", font_size=24, color=PINK)
        self.place_at_grid(bin_addr, 'D6', scale_factor=0.7)
        
        bin_breakdown = Text("011 = 001 + 010 (P1 + P2)", font_size=20, color=PINK)
        self.place_at_grid(bin_breakdown, 'E6', scale_factor=0.7)
        
        self.play(Write(bin_addr))
        self.play(Write(bin_breakdown))
        self.wait(1)

        # Final mapping visual
        # Resolved Issue 39: Set scale factor to 0.9
        math_text = Text("Position 3 = 1 + 2", font_size=32, color=YELLOW)
        self.place_in_area(math_text, 'E2', 'F5', scale_factor=0.9)
        
        self.play(Write(math_text))
        self.wait(2)
