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
        title_text = "Prerequisite: The Bernoulli Trial"
        lecture_lines = [
            "- A Bernoulli trial has exactly two outcomes.",
            "- We label success as p and failure as q.",
            "- Each dive Pixel takes is one independent trial."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Switch components
        switch_bg = RoundedRectangle(width=2.5, height=1.2, corner_radius=0.6, color=GRAY)
        self.place_in_area(switch_bg, "C3", "C4")
        
        switch_label = Text("Trial", font_size=24).next_to(switch_bg, UP, buff=0.3)
        
        success_text = Text("Success", color="#00FF00", font_size=24)
        self.place_at_grid(success_text, "C2", scale_factor=0.8) # Fixed scaling (Issue 23)
        
        failure_text = Text("Failure", color="#FF0000", font_size=24)
        self.place_at_grid(failure_text, "C5", scale_factor=0.8) # Fixed scaling (Issue 23)
        
        knob = Circle(radius=0.5, fill_opacity=1, color=WHITE)
        knob.move_to(switch_bg.get_left() + RIGHT * 0.6)
        
        self.play(
            FadeIn(switch_bg), 
            FadeIn(switch_label), 
            FadeIn(success_text), 
            FadeIn(failure_text), 
            FadeIn(knob)
        )
        
        # Toggle animation: Success to Failure and back
        self.play(knob.animate.move_to(switch_bg.get_right() - RIGHT * 0.6), run_time=1)
        self.play(knob.animate.move_to(switch_bg.get_left() + RIGHT * 0.6), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        p_label = MathTex("p", color="#ADD8E6", font_size=48)
        self.place_at_grid(p_label, "B2", scale_factor=0.7) # Fixed scaling (Issue 24)
        
        q_label = MathTex("q", color="#ADD8E6", font_size=48)
        self.place_at_grid(q_label, "B5", scale_factor=0.7) # Fixed scaling (Issue 24)
        
        self.play(Write(p_label), Write(q_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Create Pixel (using integrated asset - Issue 19)
        pixel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/penguin.svg")
        pixel.set_height(1.2) # Keeping height consistent with previous ellipse-based penguin
        self.place_at_grid(pixel, "E3")
        
        # Check and Cross
        check_mark = Text("✓", color="#00FF00", font_size=60)
        self.place_at_grid(check_mark, "E4") # Fixed position (Issue 22)
        
        cross_mark = Text("✗", color="#FF0000", font_size=60)
        self.place_at_grid(cross_mark, "E4") # Fixed position (Issue 22)

        self.play(FadeIn(pixel))
        self.wait(0.5)
        
        # Dive 1 (Success)
        self.play(pixel.animate.shift(DOWN*0.5 + RIGHT*0.5), run_time=0.6)
        self.play(knob.animate.move_to(switch_bg.get_left() + RIGHT * 0.6), run_time=0.3)
        self.play(FadeIn(check_mark))
        self.wait(1)
        self.play(FadeOut(check_mark), pixel.animate.move_to(self.grid["E3"]))
        
        # Dive 2 (Failure)
        self.play(pixel.animate.shift(DOWN*0.5 + RIGHT*0.5), run_time=0.6)
        self.play(knob.animate.move_to(switch_bg.get_right() - RIGHT * 0.6), run_time=0.3)
        self.play(FadeIn(cross_mark))
        self.wait(1)
        self.play(FadeOut(cross_mark), pixel.animate.move_to(self.grid["E3"]))
        
        self.wait(2)
        self.lecture[2].set_color(WHITE)
        self.wait(2)
