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
        self.setup_layout("Prerequisites: The Bernoulli Trial", [
            "A Bernoulli trial has two outcomes.",
            "Success and failure are defined.",
            "Probability p is always constant."
        ])
        
        # Asset Paths
        coin_icon = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg"
        dice_icon = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg"

        # === Animation for Lecture Line 1 ===
        # Create a central white circle (coin asset)
        trial_node = SVGMobject(coin_icon, color=WHITE)
        self.place_at_grid(trial_node, 'C1', scale_factor=0.3)
        self.play(FadeIn(trial_node))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Success path
        success_path = Line(self.grid['C1'], self.grid['B4'], color=GREEN)
        success_label = Text("Success", color=GREEN, font_size=20)
        self.place_at_grid(success_label, 'B4', scale_factor=0.9)
        
        # Failure path
        failure_path = Line(self.grid['C1'], self.grid['D4'], color=RED)
        failure_label = Text("Failure", color=RED, font_size=20)
        self.place_at_grid(failure_label, 'D4', scale_factor=0.9)
        
        dice_asset = SVGMobject(dice_icon)
        self.place_at_grid(dice_asset, 'E2', scale_factor=0.4)
        
        self.play(Create(success_path), Create(failure_path), Write(success_label), Write(failure_label), FadeIn(dice_asset))
        
        # Highlight branches
        self.play(success_path.animate.set_stroke(width=6), run_time=0.5)
        self.play(success_path.animate.set_stroke(width=4), run_time=0.5)
        self.play(failure_path.animate.set_stroke(width=6), run_time=0.5)
        self.play(failure_path.animate.set_stroke(width=4), run_time=0.5)
        
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        p_label = MathTex("p", color=GREEN, font_size=24)
        q_label = MathTex("q", color=RED, font_size=24)
        
        p_label.next_to(success_path.get_midpoint(), UP, buff=0.1)
        q_label.next_to(failure_path.get_midpoint(), DOWN, buff=0.1)
        
        self.play(Write(p_label), Write(q_label))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
