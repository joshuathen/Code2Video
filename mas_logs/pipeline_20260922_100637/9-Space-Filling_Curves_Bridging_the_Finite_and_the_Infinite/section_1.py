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
        self.setup_layout("The Intuitive Paradox", [
            "Lines are 1D, planes are 2D.", 
            "Can a 1D line fill a 2D space?", 
            "Intuition suggests this is impossible.", 
            "But math reveals a surprising reality.", 
            "Let us explore the space-filling curve."
        ])
        
        line_1 = Line(start=LEFT, end=RIGHT, color=WHITE)
        square_2d = Square(side_length=2, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # Addressing VideoCritic issue 20, 22
        self.place_at_grid(line_1, 'B4', scale_factor=0.8)
        self.place_in_area(square_2d, 'E4', 'F5', scale_factor=0.7)
        self.play(Create(line_1), Create(square_2d))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        # Addressing VideoCritic issue 21
        question_mark = Text("?", font_size=72, color="#FF4500")
        self.place_at_grid(question_mark, 'D3', scale_factor=0.6)
        self.play(FadeIn(question_mark))
        self.lecture[1].set_color("#FF4500")

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00CED1")
        # Asset integration
        string = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg", color="#00CED1")
        self.place_at_grid(string, 'B2', scale_factor=0.8)
        self.play(ReplacementTransform(line_1, string))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFD700")
        # Folding animation simulated with morph/highlight
        self.play(
            string.animate.set_color("#FFD700"),
            square_2d.animate.set_color("#FFD700")
        )
        self.wait(1)
