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
        self.setup_layout("Prerequisite: The Parity Bit", [
            "Parity is a simple error-detecting code.", 
            "It tracks if coin counts are even or odd.", 
            "Think of it as a binary checksum balance."
        ])
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg
        coin_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg"
        
        # Bits setup (data)
        bits = VGroup(*[SVGMobject(coin_path) for _ in range(3)])
        bits.arrange(RIGHT, buff=0.3)
        for bit in bits:
            bit.set_color(YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.place_in_area(bits, 'C3', 'C4', scale_factor=0.9)
        self.play(Create(bits))

        # === Animation for Lecture Line 2 ===
        parity_bit = SVGMobject(coin_path).set_color(RED)
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.place_at_grid(parity_bit, 'C5', scale_factor=0.9)
        self.play(FadeIn(parity_bit))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        # Visualization of a balance scale
        beam = Line(start=[-1, 0, 0], end=[1, 0, 0], color=WHITE)
        corrupted_coin = SVGMobject(coin_path).set_color(GREEN)
        self.place_at_grid(beam, 'D3', scale_factor=0.8)
        self.place_at_grid(corrupted_coin, 'D4', scale_factor=0.6)
        self.play(Create(beam), FadeIn(corrupted_coin))
