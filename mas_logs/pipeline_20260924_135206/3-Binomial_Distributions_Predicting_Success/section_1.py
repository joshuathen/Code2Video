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
        self.setup_layout("Prerequisite: The Bernoulli Trial", [
            "A Bernoulli trial has two outcomes: Success or Failure.",
            "The probability of success, p, must remain constant.",
            "Think of flipping a coin: Heads is a success."
        ])
        
        # Elements
        # Using SVGMobject as requested in instruction B025
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color="#FFD700")
        
        # Split logic
        success_part = Sector(start_angle=PI/2, angle=PI, color="#32CD32", radius=1.5)
        failure_part = Sector(start_angle=-PI/2, angle=PI, color="#FF4500", radius=1.5)
        pie_chart_group = VGroup(success_part, failure_part)
        
        success_label = Text("S", font_size=36, color="#00FF00")
        failure_label = Text("F", font_size=36, color="#FF4500")

        # === Animation for Lecture Line 1 ===
        self.place_in_area(coin, 'B2', 'E3', scale_factor=0.75)
        self.play(FadeIn(coin))
        self.wait(0.5)
        
        # Replace coin with pie chart group in the same area
        self.place_in_area(pie_chart_group, 'B2', 'E3', scale_factor=0.75)
        self.play(ReplacementTransform(coin, pie_chart_group))
        
        # Labels
        self.place_at_grid(success_label, 'B4', scale_factor=0.9)
        self.place_at_grid(failure_label, 'E4', scale_factor=0.9)
        self.play(FadeIn(success_label), FadeIn(failure_label))
        
        self.play(self.lecture[0].animate.set_color("#32CD32"))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.play(Indicate(pie_chart_group))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        self.play(Indicate(success_label), Indicate(failure_label))
        self.wait(1)
