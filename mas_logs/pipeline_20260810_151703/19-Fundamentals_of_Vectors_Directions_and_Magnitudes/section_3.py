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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Vector Addition (The Parallelogram Rule)", [
            "Connect vectors tail-to-head for addition.",
            "The resultant vector completes the parallelogram.",
            "It shows the actual path taken."
        ])
        
        # Colors for vectors
        vec_a_col = "#FF4500" # Red
        vec_b_col = "#1E90FF" # Blue
        vec_c_col = "#FFD700" # Yellow
        
        # Assets
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg").scale(0.5)
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg").scale(0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(vec_a_col)
        
        vec_a = Arrow(start=ORIGIN, end=RIGHT*1.5, color=vec_a_col, buff=0)
        vec_b = Arrow(start=ORIGIN, end=UP*1.2 + RIGHT*0.5, color=vec_b_col, buff=0)
        
        vector_group = VGroup(vec_a, vec_b)
        self.place_in_area(vector_group, 'B3', 'E5', scale_factor=0.75)
        
        self.place_at_grid(pencil, 'B2', scale_factor=0.8)
        self.play(FadeIn(pencil))
        self.play(Create(vec_a), Create(vec_b))
        self.play(FadeOut(pencil))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(vec_b_col)
        
        vec_b_shifted = vec_b.copy()
        vec_b_shifted.shift(vec_a.get_end() - vec_b.get_start())
        
        self.play(vec_b.animate.shift(vec_a.get_end() - vec_b.get_start()), run_time=1.5)
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(vec_c_col)
        
        vec_c = Arrow(start=vec_a.get_start(), end=vec_b_shifted.get_end(), color=vec_c_col, buff=0)
        
        self.place_at_grid(protractor, 'E6', scale_factor=0.8)
        self.play(FadeIn(protractor))
        self.play(Create(vec_c))
        self.play(Indicate(vec_c))
        self.play(FadeOut(protractor))
        self.wait(2)
