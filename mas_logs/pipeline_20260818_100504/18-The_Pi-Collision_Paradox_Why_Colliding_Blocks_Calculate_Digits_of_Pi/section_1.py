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
        lecture_lines = [
            "Two blocks slide toward a wall.",
            "A small block sits between wall and massive block.",
            "Exponentially heavier blocks create more collisions.",
            "Collisions count the digits of Pi.",
            "The results are 3, 31, 314, 3141."
        ]
        self.setup_layout("The Hook: The Bouncing Blocks Paradox", lecture_lines)
        
        # Load assets
        block_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        wall_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        
        block_M = block_img.copy().set_color("#FF5733")
        block_m = block_img.copy().set_color("#33FF57")
        
        label_M = Text("M", font_size=20, color=WHITE).next_to(block_M, UP)
        label_m = Text("m", font_size=20, color=WHITE).next_to(block_m, UP)
        
        group_M = VGroup(block_M, label_M)
        group_m = VGroup(block_m, label_m)
        
        wall = wall_img.scale(0.5).rotate(PI/2)
        self.place_at_grid(wall, 'D6')

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"))
        self.place_in_area(group_M, 'D2', 'D3', scale_factor=0.6)
        self.place_in_area(group_m, 'D5', 'D5', scale_factor=0.4)
        self.add(wall, group_M, group_m)
        self.play(group_M.animate.shift(RIGHT*0.5), group_m.animate.shift(LEFT*0.5))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#33FF57"))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF33"))
        self.play(group_M.animate.shift(RIGHT*1), block_M.animate.set_color("#FFFF33"))
        self.play(group_m.animate.shift(LEFT*1))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#33FF57"))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#33FF57"))
        self.wait(1)
