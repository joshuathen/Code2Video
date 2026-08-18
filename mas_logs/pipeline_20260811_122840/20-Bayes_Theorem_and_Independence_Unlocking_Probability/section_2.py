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
        lecture_lines = ["Independent events do not influence each other.", "Knowing B tells you nothing about A.", "Probabilities remain unchanged by new information."]
        self.setup_layout("The Concept of Independence", lecture_lines)
        
        # Assets
        coin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg").set_color(WHITE)
        die_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/die.svg").set_color(WHITE)
        
        # Define structures
        # Tree 1: Coin
        tree1 = VGroup(Line(ORIGIN, UP*0.8), Line(ORIGIN, DOWN*0.8)).set_color("#00FF00")
        coin = coin_icon.copy()
        
        # Tree 2: Die
        tree2 = VGroup(Line(ORIGIN, UP*0.8), Line(ORIGIN, DOWN*0.8)).set_color("#00FF00")
        die = die_icon.copy()
        
        label_a = MathTex("P(A)").set_color(WHITE).scale(0.7)
        label_not_a = MathTex("P(\\neg A)").set_color(WHITE).scale(0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        # Placing in columns 4-6 per B004
        self.place_at_grid(tree1, 'D4', scale_factor=0.8)
        self.place_at_grid(tree2, 'D6', scale_factor=0.8)
        self.place_at_grid(coin, 'B4', scale_factor=0.5)
        self.place_at_grid(die, 'B6', scale_factor=0.5)
        self.play(Create(tree1), Create(tree2), FadeIn(coin), FadeIn(die))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        # Tethering labels (B011) and using grid (C3/E3, spread out as requested)
        label_a.next_to(tree1.submobjects[0], UP, buff=0.1)
        label_not_a.next_to(tree1.submobjects[1], DOWN, buff=0.1)
        
        self.play(Write(label_a), Write(label_not_a))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFCC00"))
        # Highlight labels
        self.play(Indicate(label_a), Indicate(label_not_a))
        self.wait(2)
