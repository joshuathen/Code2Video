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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup layout
        self.setup_layout(
            "The Big Picture: Learning via Trial and Error", 
            [
                "Neural networks learn through trial and error.",
                "Every learning process starts with a simple guess.",
                "We check results and fix our mistakes."
            ]
        )
        
        # Define visual elements
        # Archer representation
        head = Circle(radius=0.2, color=WHITE)
        body = Line(DOWN*0.2, DOWN*0.6, color=WHITE)
        arms = Line(LEFT*0.3, RIGHT*0.3, color=WHITE).shift(DOWN*0.3)
        bow = Arc(radius=0.4, start_angle=-PI/2, angle=PI, color=WHITE).shift(RIGHT*0.2 + DOWN*0.3)
        archer = VGroup(head, body, arms, bow)
        
        # Target representation
        t1 = Circle(radius=0.5, color=WHITE, stroke_width=2)
        t2 = Circle(radius=0.35, color=WHITE, stroke_width=2)
        t3 = Circle(radius=0.2, color=WHITE, stroke_width=2)
        bullseye = Dot(color=WHITE, radius=0.05)
        target = VGroup(t1, t2, t3, bullseye)
        
        # Arrows
        arrow1 = Arrow(start=LEFT*0.3, end=RIGHT*0.3, color="#FFFF00", buff=0, stroke_width=3)
        arrow2 = Arrow(start=LEFT*0.3, end=RIGHT*0.3, color="#00FF00", buff=0, stroke_width=3)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Place at grid according to issue fixes (Archer D2, Target D5)
        self.place_at_grid(archer, "D2", scale_factor=0.8)
        self.place_at_grid(target, "D5", scale_factor=1.0)
        
        self.play(FadeIn(archer), FadeIn(target))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        
        # Shoot first arrow (miss high)
        miss_pos = self.grid["B5"]
        v_miss = miss_pos - self.grid["D2"]
        theta_miss = np.arctan2(v_miss[1], v_miss[0])
        
        arrow1.move_to(self.grid["D2"])
        arrow1.rotate(theta_miss)
        
        self.play(arrow1.animate.move_to(miss_pos), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#FF0000")
        )
        
        # Bracket measuring distance from bullseye (D5) to arrow (B5)
        bracket = BraceBetweenPoints(self.grid["D5"], self.grid["B5"], direction=RIGHT, color="#FF0000")
        dist_label = Text("Error", font_size=18, color="#FF0000").next_to(bracket, RIGHT, buff=0.1)
        
        self.play(Create(bracket), Write(dist_label))
        self.wait(1)
        
        # Archer tilts/adjusts downwards
        self.play(
            archer.animate.rotate(-0.3, about_point=self.grid["D2"]),
            FadeOut(bracket),
            FadeOut(dist_label),
            FadeOut(arrow1)
        )
        
        # Shoot second arrow (hit bullseye)
        hit_pos = self.grid["D5"]
        v_hit = hit_pos - self.grid["D2"]
        theta_hit = np.arctan2(v_hit[1], v_hit[0])
        
        arrow2.move_to(self.grid["D2"])
        arrow2.rotate(theta_hit)
        
        self.play(arrow2.animate.move_to(hit_pos), run_time=1.5)
        self.wait(2)
