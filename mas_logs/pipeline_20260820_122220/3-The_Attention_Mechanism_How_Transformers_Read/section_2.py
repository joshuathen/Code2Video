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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Queries, Keys, and Values drive the mechanism.",
            "Query: What am I looking for?",
            "Key: What do I have to offer?",
            "Value: The content I provide.",
            "Dot products measure similarity between vectors."
        ]
        self.setup_layout("The Mechanism: Queries, Keys, and Values", lecture_lines)
        
        # Colors: Q (#FF0000), K (#00FF00), V (#0000FF), Score (#FFFF00), Base (#FFFFFF)
        c_q = "#FF0000"
        c_k = "#00FF00"
        c_v = "#0000FF"
        c_score = "#FFFF00"
        
        # Objects
        q_vec = Arrow(start=ORIGIN, end=RIGHT*1, color=c_q, buff=0)
        k_vec = Arrow(start=ORIGIN, end=UP*1, color=c_k, buff=0)
        v_vec = Arrow(start=ORIGIN, end=RIGHT+UP, color=c_v, buff=0)
        
        q_label = Text("Q", font_size=24, color=c_q).scale(0.7)
        k_label = Text("K", font_size=24, color=c_k).scale(0.7)
        v_label = Text("V", font_size=24, color=c_v).scale(0.7)
        
        # Tethering labels (B011)
        q_label.next_to(q_vec, RIGHT, buff=0.1)
        k_label.next_to(k_vec, UP, buff=0.1)
        v_label.next_to(v_vec, UR, buff=0.1)
        
        vector_diagram = VGroup(q_vec, k_vec, v_vec, q_label, k_label, v_label)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.place_in_area(vector_diagram, 'B4', 'E6', scale_factor=0.8)
        self.play(Create(q_vec), Write(q_label), Create(k_vec), Write(k_label), Create(v_vec), Write(v_label))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(c_q)
        self.play(Indicate(q_vec))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(c_k)
        self.play(Indicate(k_vec))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(c_v)
        self.play(Indicate(v_vec))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        score = Dot(color=c_score).move_to(k_vec.get_end())
        self.play(Create(score))
        self.wait(2)
