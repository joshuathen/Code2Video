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
        lecture_lines = ["A Bernoulli trial has only two possible outcomes.", 
                         "Outcomes must be either success or failure.", 
                         "Each trial maintains a constant success probability."]
        self.setup_layout("Prerequisite: The Bernoulli Trial", lecture_lines)
        
        # Elements
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color=WHITE)
        success_label = Text("Success", font_size=20, color=GREEN)
        failure_label = Text("Failure", font_size=20, color=RED)
        trial_label = Text("Trial", font_size=24, color=WHITE)
        
        coin_group = VGroup(coin, success_label, failure_label, trial_label)
        success_label.next_to(coin, UP, buff=0.1)
        failure_label.next_to(coin, DOWN, buff=0.1)
        trial_label.next_to(coin, RIGHT, buff=0.1)
        
        self.place_in_area(coin_group, 'C4', 'E6', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(coin_group))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Indicate(success_label, color=YELLOW))
        self.lecture[1].set_color(YELLOW)
        self.wait(0.5)
        self.play(Indicate(failure_label, color=GREY))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.play(coin.animate.set_stroke(color=YELLOW, width=4), run_time=1)
        self.wait(1)
