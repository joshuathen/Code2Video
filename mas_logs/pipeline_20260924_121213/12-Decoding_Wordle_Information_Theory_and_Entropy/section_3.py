from manim import *
import numpy as np

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
        self.setup_layout("Applying Entropy to Wordle", [
            "Every guess partitions the remaining word list.",
            "High-entropy guesses create more balanced outcomes.",
            "Visualizing as a branching tree helps.",
            "Good guesses split candidates into similar sizes.",
            "Bad guesses leave one huge remaining set."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display the entropy formula on a monitor icon. (#FFFFFF)
        monitor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/monitor.svg", color="#FFFFFF")
        formula = MathTex("H = -\\sum p \\log p", color="#FFFFFF")
        group1 = VGroup(monitor, formula).arrange(DOWN)
        self.place_in_area(group1, 'A2', 'C4', scale_factor=0.8)
        self.play(FadeIn(monitor), Write(formula))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show probability distribution on a computer icon. (#FF00FF)
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color="#FF00FF")
        dist = BarChart([5, 10, 20, 10, 5], bar_colors=["#FF00FF"]*5).scale(0.5)
        group2 = VGroup(computer, dist).arrange(DOWN)
        self.place_in_area(group2, 'D2', 'F4', scale_factor=0.7)
        self.play(FadeIn(computer), FadeIn(dist))
        self.lecture[1].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight high-entropy words with a bright outline. (#FFFF00)
        box = Rectangle(width=2, height=1, color="#FFFF00")
        self.place_at_grid(box, "B5", scale_factor=0.8)
        self.play(Create(box))
        self.lecture[2].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Demonstrate uncertainty for guesses using a keyboard icon. (#00FFFF)
        keyboard = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/keyboard.svg", color="#00FFFF")
        label = Text("Uncertainty: Low", color="#00FFFF", font_size=20)
        group4 = VGroup(keyboard, label).arrange(DOWN)
        self.place_at_grid(group4, 'E5', scale_factor=0.7)
        self.play(FadeIn(keyboard), Write(label))
        self.lecture[3].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Summarize with a decision model using a calculator icon. (#FFFFFF)
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg", color="#FFFFFF")
        summary = Text("Decision Model", color="#FFFFFF", font_size=20)
        group5 = VGroup(calculator, summary).arrange(DOWN)
        self.place_at_grid(group5, 'F6', scale_factor=0.7)
        self.play(FadeIn(calculator), Write(summary))
        self.lecture[4].set_color("#FFFFFF")
        self.wait(1)
