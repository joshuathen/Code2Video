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
        self.setup_layout("Prerequisite Review: Bernoulli Trials", ["A Bernoulli trial has two outcomes.", "Success happens with probability p.", "Failure occurs with 1 minus p."])
        
        # === Animation for Lecture Line 1 ===
        # Bernoulli Trial definition with die.svg
        die_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/die.svg", color=WHITE)
        trial_label = Text("Trial", font_size=24)
        trial_group = VGroup(die_icon, trial_label).arrange(DOWN)
        self.place_at_grid(trial_group, 'C5', scale_factor=0.7)
        self.play(FadeIn(trial_group))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Show success (p)
        success_label = Text("Success (p)", font_size=24, color="#FFFF00")
        self.place_at_grid(success_label, 'B5', scale_factor=0.8)
        self.play(Write(success_label))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Show failure (q=1-p) with coin.svg
        coin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color="#00FF00")
        failure_label = Text("Failure (1-p)", font_size=24, color="#FFFF00")
        fail_group = VGroup(failure_label, coin_icon).arrange(DOWN)
        self.place_at_grid(fail_group, 'D5', scale_factor=0.6)
        self.play(Write(failure_label), FadeIn(coin_icon))
        self.lecture[2].set_color("#FFFF00")
        
        self.wait(2)
