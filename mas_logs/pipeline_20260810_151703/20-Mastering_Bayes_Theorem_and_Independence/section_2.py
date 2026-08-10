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
        lecture_lines = ["Independence means knowing B gives no info on A.", "The probability of A remains unchanged by B.", "Visually, the ratio of A is constant throughout."]
        self.setup_layout("The Concept of Independence", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show two coin flip outcomes using asset
        coin1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color=WHITE)
        coin2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color=WHITE)
        coins = VGroup(coin1, coin2).arrange(RIGHT, buff=0.5)
        self.place_at_grid(coins, 'B4', scale_factor=0.8)
        self.play(Create(coins))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display P(A|B) = P(A) formula
        formula = MathTex("P(A|B)", "=", "P(A)", font_size=36)
        self.place_in_area(formula, 'D3', 'E5', scale_factor=0.9)
        self.play(Write(formula))
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate ratio of A remaining constant despite change in B using assets
        bar_bg = Rectangle(height=2, width=3, color=GRAY, fill_opacity=0.2)
        bar_a = Rectangle(height=1, width=3, color=BLUE, fill_opacity=0.8).align_to(bar_bg, DOWN)
        ratio_viz = VGroup(bar_bg, bar_a)
        
        # Using asset as requested in animation 3 (removed invalid 'scale' kwarg)
        coin3 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        self.place_in_area(ratio_viz, 'B5', 'C6', scale_factor=0.5)
        self.place_at_grid(coin3, 'C6', scale_factor=0.3)
        
        self.play(Create(ratio_viz), Create(coin3))
        # "Change" in B visually represented by shifting colors/movement
        self.play(bar_a.animate.set_color(GREEN), run_time=1)
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
