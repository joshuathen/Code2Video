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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Mechanism: Decoding the Secret", [
            "The second prisoner calculates the total XOR sum.",
            "The result reveals the specific coin changed.",
            "The prisoners' secret strategy guarantees their escape."
        ])
        
        # Assets
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        coin_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        
        # Place assets in defined area
        self.place_in_area(grid_asset, "A2", "F6", scale_factor=0.5)
        self.place_in_area(coin_asset, "C3", "D4", scale_factor=0.3)
        
        # === Animation for Lecture Line 1 ===
        # The second prisoner calculates the total XOR sum.
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Create trace path on grid
        trace_path = VGroup()
        for i in range(1, 7):
            trace_path.add(Line(self.grid[f"A{i}"], self.grid[f"F{i}"], color="#00FF00"))
        
        self.play(FadeIn(grid_asset), Create(trace_path), run_time=2)
        
        # === Animation for Lecture Line 2 ===
        # The result reveals the specific coin changed.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Show Coin
        self.play(FadeIn(coin_asset))
        
        # === Animation for Lecture Line 3 ===
        # The prisoners' secret strategy guarantees their escape.
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        
        # Final flourish - surrounding highlight
        highlight = SurroundingRectangle(coin_asset, color="#FF00FF", buff=0.1)
        self.play(Create(highlight))
        
        self.wait(2)
