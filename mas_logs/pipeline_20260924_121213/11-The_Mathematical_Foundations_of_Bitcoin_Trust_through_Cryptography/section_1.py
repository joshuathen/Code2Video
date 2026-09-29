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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Digital money faces the double-spending problem.",
            "Users send coins to multiple parties simultaneously.",
            "We need a decentralized way to ensure uniqueness."
        ]
        self.setup_layout("Introduction: The Digital Ledger Problem", lecture_lines)
        
        # Load assets
        ledger = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ledger.svg")
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        
        # Create transaction visuals
        tx1 = Text("Tx: Alice -> Bob", font_size=20, color=BLUE)
        tx2 = Text("Tx: Alice -> Charlie", font_size=20, color=BLUE)
        tx_list = VGroup(ledger, tx1, tx2).arrange(DOWN, buff=0.3)
        self.place_in_area(tx_list, "A3", "C5", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(FadeIn(tx_list))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        coin.set_color(RED)
        self.place_at_grid(coin, "C5", scale_factor=0.5)
        self.play(FadeIn(coin))
        
        # Highlight double spend (red)
        red_box = SurroundingRectangle(tx_list, color=RED, buff=0.2)
        self.play(Create(red_box))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        trust_text = Text("Trust Problem", font_size=32, color=YELLOW)
        self.place_at_grid(trust_text, "D3", scale_factor=0.8)
        self.play(Write(trust_text))
        self.wait(2)
