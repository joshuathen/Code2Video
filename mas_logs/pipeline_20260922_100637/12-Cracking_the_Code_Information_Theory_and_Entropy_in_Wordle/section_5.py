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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Entropy is the secret behind efficient data compression.",
            "It powers machine learning models and decision trees.",
            "Your smartphone keyboard predicts words using these principles."
        ]
        self.setup_layout("Conclusion: Beyond Games", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Entropy applies everywhere
        text_label = Text("Entropy applies everywhere", font_size=32, color=WHITE)
        self.place_at_grid(text_label, 'B2')
        self.play(FadeIn(text_label))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Icons for stocks and DNA
        stock_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stock.svg")
        dna_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dna.svg")
        
        # Fixing clutter issues per Critic:
        # 1. Use separate objects as required
        dot_icon = VGroup(stock_icon, dna_icon).arrange(RIGHT, buff=0.5)
        self.place_at_grid(dot_icon, 'B5', scale_factor=0.6)
        
        labels_group = VGroup(
            Text("Stocks", font_size=18, color="#FFD700"),
            Text("DNA", font_size=18, color="#FFD700")
        ).arrange(RIGHT, buff=0.5)
        self.place_in_area(labels_group, 'B6', 'C6', scale_factor=0.5)

        # 3. grid_visual (optional placeholder for context)
        grid_visual = Rectangle(width=2, height=1, color=GRAY)
        self.place_in_area(grid_visual, 'D3', 'F6', scale_factor=0.6)
        
        self.play(Create(dot_icon), Write(labels_group), Create(grid_visual))
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        self.wait(2)
        self.lecture[2].set_color("#FFD700")
        self.play(FadeOut(text_label), FadeOut(dot_icon), FadeOut(labels_group), FadeOut(grid_visual), run_time=1.5)
