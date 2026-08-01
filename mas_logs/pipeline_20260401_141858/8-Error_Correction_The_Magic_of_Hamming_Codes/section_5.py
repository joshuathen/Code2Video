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

class Section5Scene(TeachingScene):
    def construct(self):
        # Define lecture lines
        lecture_lines = [
            "We calculate a syndrome by checking every parity circle.",
            "Unbalanced circles reveal where the parity rule is broken.",
            "The intersection of broken circles identifies the faulty bit.",
            "If circles B and C fail, position six is wrong.",
            "Flip the bit back to fix the corrupted data."
        ]
        
        self.setup_layout("The Error Correction Process", lecture_lines)

        # Colors
        COLOR_A = BLUE_D
        COLOR_B = PURPLE_D
        COLOR_C = ORANGE
        COLOR_PASS = "#00FF00"
        COLOR_FAIL = "#FF0000"
        COLOR_HIGHLIGHT = YELLOW

        # Bits for sequence 1110101 (Storyboard target)
        # Position mapping for Venn: 1, 2, 3, 4, 5, 6, 7
        bits_values = ["1", "1", "1", "0", "1", "1", "1"]
        pos_map = ["A4", "E2", "C3", "E6", "C5", "E4", "D4"]
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Asset: Data icon
        data_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/data.svg")
        self.place_at_grid(data_icon, "A2", scale_factor=0.4)
        
        # Create Venn Circles (Shifted Up as per Critic Fix)
        circle_a = Circle(radius=1.4, color=COLOR_A, stroke_width=4)
        circle_b = Circle(radius=1.4, color=COLOR_B, stroke_width=4)
        circle_c = Circle(radius=1.4, color=COLOR_C, stroke_width=4)

        self.place_at_grid(circle_a, "B4")
        self.place_at_grid(circle_b, "D3")
        self.place_at_grid(circle_c, "D5")
        
        labels = VGroup(
            Text("A", font_size=24, color=COLOR_A).next_to(circle_a, UP, buff=0.1),
            Text("B", font_size=24, color=COLOR_B).next_to(circle_b, DOWN + LEFT, buff=0.1),
            Text("C", font_size=24, color=COLOR_C).next_to(circle_c, DOWN + RIGHT, buff=0.1)
        )

        # Create Bits
        bits_mobjects = VGroup()
        for val, grid_pos in zip(bits_values, pos_map):
            b_text = Text(val, font_size=32, color=WHITE)
            self.place_at_grid(b_text, grid_pos)
            bits_mobjects.add(b_text)

        self.play(FadeIn(data_icon), Create(circle_a), Create(circle_b), Create(circle_c), FadeIn(labels))
        self.play(FadeIn(bits_mobjects))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        # Parity Check Circle A (Glows Green)
        self.play(circle_a.animate.set_stroke(color=COLOR_PASS, width=10))
        self.wait(0.5)
        self.play(circle_a.animate.set_stroke(width=4))
        
        # Parity Check Circle B & C (Glow Red)
        self.play(
            circle_b.animate.set_stroke(color=COLOR_FAIL, width=10),
            circle_c.animate.set_stroke(color=COLOR_FAIL, width=10)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        # Highlight intersection region (Position 6) (Shifted Up as per Critic Fix)
        highlight_box = Square(side_length=0.7, color=COLOR_HIGHLIGHT).set_stroke(width=6)
        self.place_at_grid(highlight_box, "E4")
        
        self.play(Create(highlight_box))
        self.play(bits_mobjects[5].animate.set_color(COLOR_HIGHLIGHT).scale(1.2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        
        # Error Label (Scaled as per Critic Fix)
        error_label = Text("Error Detected!", font_size=22, color=COLOR_FAIL)
        self.place_at_grid(error_label, "F5", scale_factor=0.7)
        self.play(Write(error_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        
        # Flip Bit 6: 1 -> 0 (Shifted Up as per Critic Fix)
        new_bit_6 = Text("0", font_size=32, color=COLOR_PASS)
        self.place_at_grid(new_bit_6, "E4")
        
        # Confirmation icon update
        data_icon_fixed = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/data.svg").set_color(COLOR_PASS)
        self.place_at_grid(data_icon_fixed, "A2", scale_factor=0.4)

        self.play(
            FadeOut(error_label), 
            ReplacementTransform(bits_mobjects[5], new_bit_6),
            ReplacementTransform(data_icon, data_icon_fixed)
        )
        self.play(Uncreate(highlight_box))
        
        # Show all circles turning green
        self.play(
            circle_b.animate.set_stroke(color=COLOR_PASS, width=4),
            circle_c.animate.set_stroke(color=COLOR_PASS, width=4),
            circle_a.animate.set_stroke(color=COLOR_PASS, width=4)
        )
        
        # Final bit sequence glow
        all_bits = VGroup(*bits_mobjects[:5], new_bit_6, bits_mobjects[6])
        self.play(all_bits.animate.set_color(COLOR_PASS))
        self.wait(2)
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
