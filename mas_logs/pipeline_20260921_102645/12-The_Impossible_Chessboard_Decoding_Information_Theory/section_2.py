from manim import *

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
        self.setup_layout("Prerequisite: The Parity Principle", [
            "Parity tracks the state of a sequence.",
            "XOR operations determine if parity is even.",
            "Flipping one coin always reverses the parity."
        ])
        
        # Binary string: 1 0 1 1 0
        bits_str = "1 0 1 1 0"
        bits = VGroup(*[Text(c, font_size=36, color="#00FFFF") for c in bits_str.split()])
        bits.arrange(RIGHT, buff=0.4)
        
        # Add Asset
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        coin.scale(0.3)
        coin_group = VGroup(bits, coin).arrange(RIGHT, buff=0.5)
        
        # Fix 24: Move bits/coin up
        self.place_at_grid(coin_group, "B3", scale_factor=0.8)
        
        bits_label = Text("Bits:", font_size=24, color=WHITE)
        bits_label.next_to(coin_group, UP, buff=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"), Write(bits), Write(coin), Write(bits_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        xor_text = Text("1 ⊕ 0 ⊕ 1 ⊕ 1 ⊕ 0 = 1", font_size=28, color="#00FF00")
        # Fix 25: Move xor_text
        self.place_at_grid(xor_text, "C3", scale_factor=0.8)
        self.play(FadeIn(xor_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        
        # Highlight last bit and flip
        target_bit = bits[-1]
        self.play(Indicate(target_bit, color="#FF00FF"))
        
        new_bit = Text("1", font_size=36, color="#FF00FF") # Should be "0" -> "1"
        new_bit.move_to(target_bit)
        
        self.play(Transform(target_bit, new_bit))
        
        final_xor = Text("1 ⊕ 0 ⊕ 1 ⊕ 1 ⊕ 1 = 0", font_size=28, color="#FF00FF")
        # Fix 26: Move final_xor
        self.place_at_grid(final_xor, "D3", scale_factor=0.8)
        self.play(ReplacementTransform(xor_text, final_xor))
        
        self.wait(2)
