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
        self.setup_layout("The Hook: How many pieces?", [
            "Place points on a circle.", 
            "Connect every pair with chords.", 
            "How many regions are created?", 
            "One point gives one region.", 
            "Two points give two regions."
        ])
        
        # Elements using assets
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=WHITE)
        point1 = Dot(color=YELLOW)
        point2 = Dot(color=YELLOW)
        label_p1 = Text("P1", font_size=18, color=YELLOW)
        label_p2 = Text("P2", font_size=18, color=YELLOW)
        chord = Line(start=point1.get_center(), end=point2.get_center(), color="#00FFFF")
        
        # Placement logic
        self.place_at_grid(circle, 'D4', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(circle), self.lecture[0].animate.set_color(YELLOW))
        
        # P1 on circumference
        point1.move_to(circle.get_center() + 0.6 * np.array([np.cos(PI/2), np.sin(PI/2), 0]))
        label_p1.next_to(point1, UP, buff=0.1)
        self.play(FadeIn(point1), Write(label_p1))
        
        # === Animation for Lecture Line 2 ===
        # P2 on circumference
        point2.move_to(circle.get_center() + 0.6 * np.array([np.cos(-PI/2), np.sin(-PI/2), 0]))
        label_p2.next_to(point2, DOWN, buff=0.1)
        chord.put_start_and_end_on(point1.get_center(), point2.get_center())
        
        self.play(FadeIn(point2), Write(label_p2), Create(chord), self.lecture[1].animate.set_color(GOLD))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE))
        
        # === Animation for Lecture Line 5 ===
        summary_text = Text("1 point = 1 region; 2 points = 2 regions", font_size=20, color=YELLOW)
        self.place_in_area(summary_text, 'B1', 'D3', scale_factor=0.65)
        self.play(Write(summary_text), self.lecture[4].animate.set_color(WHITE))
        
        self.wait(1)
