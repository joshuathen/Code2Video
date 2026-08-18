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
        lines = ["Consider two blocks on a frictionless track.", "One block hits another and a wall.", "We count the total collisions between objects."]
        self.setup_layout("The Setup: A Curious Collision", lines)
        
        # Mobjects
        block1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=WHITE)
        block2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=WHITE)
        wall = Line(start=UP*1.5, end=DOWN*1.5, color=GREY, stroke_width=10)
        
        label1 = Text("block_1", font_size=20, color=WHITE).scale(0.75)
        label2 = Text("block_2", font_size=20, color=WHITE).scale(0.75)

        self.place_at_grid(block1, 'D4', scale_factor=0.8)
        self.place_at_grid(block2, 'D5', scale_factor=0.8)
        self.place_in_area(wall, 'B6', 'E6', scale_factor=0.7)
        
        label1.next_to(block1, UP, buff=0.1)
        label2.next_to(block2, UP, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.add(block1, block2, wall, label1, label2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        self.play(block1.animate.set_color("#FF00FF"), label1.animate.set_color("#FF00FF"))
        self.play(block1.animate.move_to(self.grid['D5'] + LEFT*0.7), run_time=1)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        # Collision simulation
        self.play(block1.animate.move_to(self.grid['D5'] + LEFT*0.1), run_time=0.3)
        self.play(block1.animate.move_to(self.grid['D5'] + LEFT*0.7), run_time=0.3)
        self.wait(1)
