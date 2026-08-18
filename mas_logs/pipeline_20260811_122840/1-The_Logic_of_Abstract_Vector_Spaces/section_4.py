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
        self.setup_layout("Application: The Geometry of Data", [
            "Data lives in feature spaces.",
            "Emojis act as coordinate vectors.",
            "Adding moods creates new feelings."
        ])

        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/emoji.svg]
        emoji = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/emoji.svg")
        data_points = VGroup(*[emoji.copy() for _ in range(8)])
        for p in data_points:
            # Using Row B to E, Cols 4 to 6 (B004, B008)
            grid_pos = f"{np.random.choice(['B', 'C', 'D', 'E'])}{np.random.choice(['4', '5', '6'])}"
            self.place_at_grid(p, grid_pos, scale_factor=0.2)
        
        self.play(FadeIn(data_points))
        self.lecture[0].set_color("#8A2BE2")

        # === Animation for Lecture Line 2 ===
        # Line of best fit
        line = Line(start=self.grid['B4'], end=self.grid['E6'], color=WHITE)
        self.play(Create(line))
        self.lecture[1].set_color(WHITE)

        # === Animation for Lecture Line 3 ===
        projections = VGroup()
        for p in data_points:
            # Projection logic
            start_vec = line.get_start()
            line_vec = line.get_vector()
            p_vec = p.get_center()
            t = np.dot(p_vec - start_vec, line_vec) / np.dot(line_vec, line_vec)
            proj = start_vec + np.clip(t, 0, 1) * line_vec
            projections.add(Line(p_vec, proj, color=YELLOW, stroke_width=2))
        
        self.play(Create(projections))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
