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
        self.setup_layout("The Intuition: What is Information?", [
            "Information is the reduction of uncertainty.",
            "Surprise drives how much we learn.",
            "Binary splits provide exactly one bit."
        ])
        
        # Assets
        switch_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg")
        coin_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")

        # === Animation for Lecture Line 1 ===
        # Visualize a single binary choice (Yes/No) turning into certainty using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg]. #FFFFFF
        uncertainty_circle = Circle(radius=0.5, color=WHITE)
        self.place_at_grid(uncertainty_circle, 'B5', scale_factor=0.6)
        
        self.place_at_grid(switch_asset, 'B5', scale_factor=0.3)
        self.add(uncertainty_circle, switch_asset)
        
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Fade in a probability distribution changing from uniform to spike. #00FF00
        rectangle_blocks = VGroup(*[Rectangle(height=0.8, width=0.4, color=GREY, fill_opacity=0.5) for _ in range(4)])
        self.place_in_area(rectangle_blocks, 'D4', 'F6', scale_factor=0.7)
        self.add(rectangle_blocks)
        
        self.lecture[1].set_color("#00FF00")
        
        spike = Rectangle(height=1.5, width=0.4, color="#00FF00", fill_opacity=1.0)
        self.place_at_grid(spike, 'E5', scale_factor=0.7)
        self.play(Transform(rectangle_blocks, spike))
        
        # === Animation for Lecture Line 3 ===
        # Highlight the reduction in possible outcomes as information arrives, represented by a accumulating [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg]. #FFFF00
        arrow = Arrow(start=self.grid['D1'], end=self.grid['D3'], color=WHITE)
        self.place_at_grid(arrow, 'D2', scale_factor=0.5)
        self.play(Create(arrow))
        
        self.lecture[2].set_color("#FFFF00")
        
        self.place_at_grid(coin_asset, 'C2', scale_factor=0.3)
        self.play(FadeIn(coin_asset))
