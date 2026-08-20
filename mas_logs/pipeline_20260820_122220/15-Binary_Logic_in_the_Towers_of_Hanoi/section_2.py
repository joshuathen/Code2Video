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
        lecture_lines = ["The puzzle has rods and disks.", "Move disks without breaking the rules.", "Three disks require seven moves."]
        self.setup_layout("The Towers of Hanoi Puzzle", lecture_lines)
        
        # Load assets
        rod = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg", color=GRAY)
        disk = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")
        
        # Create rods and disks
        rods = VGroup(*[rod.copy() for _ in range(3)])
        disks = VGroup(*[disk.copy().set_color(color) for color in [RED, GREEN, BLUE]])
        
        # Arrange rods
        rods.arrange(RIGHT, buff=1.5)
        
        # Initial stack on the first rod (A4-F6 area as per recommendation 25)
        stack = VGroup(*disks).arrange(DOWN, buff=0.0)
        stack.move_to(rods[0].get_center() + DOWN * 0.5)
        
        hanoi_group = VGroup(rods, stack)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_in_area(hanoi_group, "A4", "F6", scale_factor=0.6)
        self.add(hanoi_group)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        # Shift smallest disk to target rod
        self.play(stack[2].animate.move_to(rods[2].get_center() + DOWN * 0.5))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00BFFF")
        # Shift middle disk to middle rod
        self.play(stack[1].animate.move_to(rods[1].get_center() + DOWN * 0.5))
        self.wait(2)
