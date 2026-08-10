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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Putting It Together: A Real-World Scenario", [
            "Robot R2 scans for a gold coin in a grid.",
            "The prior probability the coin exists is point-one.",
            "R2 is ninety percent accurate in its detection.",
            "If R2 beeps, we apply Bayes' to update odds.",
            "Prior knowledge is crucial for accurate real-world inference."
        ])
        
        # Assets
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color="#FFD700")
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", color="#00FFFF")
        sensor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sensor.svg", color="#00FFFF")
        
        # Group assets
        visual_group = VGroup(coin, robot, sensor)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#0000FF") # BLUE
        self.place_at_grid(coin, 'C3', scale_factor=0.5)
        self.play(FadeIn(coin))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00") # GREEN
        prior_text = Text("P(Coin)=0.1", font_size=20, color="#FFD700")
        self.place_at_grid(prior_text, 'C4')
        self.play(Write(prior_text))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00") # YELLOW
        self.place_at_grid(robot, 'D3', scale_factor=0.5)
        self.place_at_grid(sensor, 'D4', scale_factor=0.5)
        self.play(FadeIn(robot), FadeIn(sensor))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFA500") # ORANGE
        final_val = MathTex("P_{updated} > 0.1", color=WHITE)
        self.place_at_grid(final_val, 'D4', scale_factor=0.8)
        self.play(Write(final_val))
        self.play(Flash(final_val))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#800080") # PURPLE
        self.play(FadeOut(visual_group), FadeOut(prior_text), FadeOut(final_val))
