from manim import *
import os

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
        self.setup_layout("Visualizing Attention Scores (Dot-Product)", [
            "Dot-products measure word relevance.", 
            "Softmax converts scores into weights.", 
            "High relevance creates a spotlight."
        ])
        
        self.lecture.set_opacity(1)

        # Animation Setup
        vec_q = Vector([1, 0], color=YELLOW)
        vec_k = Vector([0.5, 0.8], color=BLUE)
        self.place_at_grid(vec_q, "B2", scale_factor=0.8)
        self.place_at_grid(vec_k, "B2", scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(vec_q), FadeIn(vec_k))
        self.play(Rotate(vec_k, angle=-0.5), run_time=1.5)
        
        # Projection line
        proj_val = 0.7
        proj_line = DashedLine(vec_k.get_end(), [vec_k.get_end()[0], vec_q.get_start()[1], 0], color=PURPLE)
        self.play(Create(proj_line))
        
        val_text = MathTex("Score = 0.7", color="#FF00FF")
        self.place_at_grid(val_text, "D2", scale_factor=0.7)
        self.play(Write(val_text))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        
        softmax_text = Text("Softmax(Score) -> 0.85", font_size=24, color=BLUE)
        self.place_at_grid(softmax_text, "E2", scale_factor=0.7)
        self.play(FadeIn(softmax_text))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GOLD)
        
        # Use SVG asset as requested
        spotlight = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spotlight.svg")
        spotlight.set_color(GOLD)
        self.place_at_grid(spotlight, "C3", scale_factor=0.8)
        self.play(FadeIn(spotlight))
        self.wait(2)
