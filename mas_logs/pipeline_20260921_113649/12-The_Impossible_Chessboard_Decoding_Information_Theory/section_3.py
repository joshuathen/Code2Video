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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Alice calculates the parity of heads.",
            "She flips a coin to change parity.",
            "The parity now points to the target.",
            "Bob computes the board's new parity.",
            "He correctly identifies the target square."
        ]
        self.setup_layout("The Solution: Parity Strategy", lecture_lines)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg]
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg]
        coin_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        
        coin_grid = VGroup()
        for i in range(4):
            for j in range(4):
                coin = coin_asset.copy()
                coin.scale(0.3)
                coin.move_to(self.grid[f"{chr(67+i)}{chr(51+j)}"])
                coin_grid.add(coin)
        
        self.place_in_area(grid_asset, 'A4', 'F6', scale_factor=0.9)
        self.place_in_area(coin_grid, 'A4', 'F6', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(grid_asset), FadeIn(coin_grid))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(ORANGE))
        target_coin = coin_grid[0]
        self.play(target_coin.animate.set_color(RED))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        arc_arrow = CurvedArrow(self.grid['B4'], self.grid['F6'], color=WHITE)
        self.place_at_grid(arc_arrow, 'D4', scale_factor=1.0)
        self.play(Create(arc_arrow))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(BLUE))
        self.play(Indicate(coin_grid))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(RED))
        target_square = coin_grid[15]
        target_square.set_stroke(color=RED, width=5)
        self.play(Flash(target_square.get_center(), color=RED))
