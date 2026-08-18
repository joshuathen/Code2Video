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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Null space collapses vectors to zero.", "It acts as a ghost zone.", "Everything here becomes invisible to transformation."]
        self.setup_layout("Null Space: The 'Ghost' Zone", lecture_lines)
        
        # Define elements
        dot = Dot(color=BLUE)
        origin = Dot(color=RED)
        line = Line(start=LEFT*1, end=RIGHT*1, color=YELLOW)
        area = Rectangle(width=4, height=0.5, color=GREEN, fill_opacity=0.3)
        ghost = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ghost.svg")
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(dot, 'B3', scale_factor=0.8)
        self.place_at_grid(origin, 'B5', scale_factor=0.8)
        self.play(FadeIn(dot), FadeIn(origin))
        self.play(dot.animate.move_to(origin.get_center()))
        self.lecture[0].set_color("#FF0000")

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(line, 'D4', scale_factor=0.9)
        self.play(Create(line))
        self.lecture[1].set_color("#0000FF")

        # === Animation for Lecture Line 3 ===
        self.place_in_area(area, 'D3', 'D5', scale_factor=0.8)
        self.play(FadeIn(area))
        
        # Bring ghost into area
        self.place_at_grid(ghost, 'D4', scale_factor=0.5)
        self.play(FadeIn(ghost), FadeOut(dot), FadeOut(line))
        
        # Shrink until vanishes
        self.play(ghost.animate.scale(0.01), run_time=1.5)
        self.play(FadeOut(ghost))
        
        self.lecture[2].set_color("#00FF00")
        
        self.wait(2)
