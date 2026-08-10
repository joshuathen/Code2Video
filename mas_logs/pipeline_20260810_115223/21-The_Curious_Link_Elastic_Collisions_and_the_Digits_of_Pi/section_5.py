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
        self.setup_layout("Conclusion and Intuition", [
            "Nature hides constants in chaotic systems.",
            "Conservation laws reveal deep mathematical truths.",
            "Physics effectively calculates fundamental numbers."
        ])
        
        # === Animation for Lecture Line 1 ===
        sum_text = Text("Conservation laws can calculate digits of π.", color=WHITE, font_size=24)
        self.place_in_area(sum_text, 'A3', 'B6', scale_factor=0.6)
        self.play(Write(sum_text))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Sumo-Hamster visual simulation using Assets
        sumo = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sumo.svg")
        hamster = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hamster.svg")
        
        sumo_label = Text("M", font_size=20)
        hamster_label = Text("m", font_size=20)
        
        sumo_group = VGroup(sumo, sumo_label).arrange(DOWN)
        hamster_group = VGroup(hamster, hamster_label).arrange(DOWN)
        
        simulation = VGroup(sumo_group, hamster_group).arrange(RIGHT, buff=1.0)
        self.place_at_grid(simulation, 'D5', scale_factor=0.7)
        
        self.play(FadeIn(simulation))
        # Simulate bounces
        self.play(
            hamster_group.animate.shift(LEFT * 0.5),
            hamster_group.animate.shift(RIGHT * 0.5),
            run_time=2
        )
        self.lecture[1].set_color(BLUE)

        # === Animation for Lecture Line 3 ===
        conclusion = Text("Mathematics hidden in chaos", color="#ADD8E6", font_size=32, weight=BOLD)
        self.place_at_grid(conclusion, 'E5', scale_factor=0.7)
        self.play(FadeIn(conclusion))
        self.lecture[2].set_color(BLUE)
        self.wait(2)
