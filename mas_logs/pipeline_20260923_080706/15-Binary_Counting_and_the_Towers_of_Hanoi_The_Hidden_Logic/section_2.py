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
        self.setup_layout("The Towers of Hanoi: Rules of the Game", [
            "Three rods hold disks of different sizes.",
            "Rule: Move one disk at a time.",
            "Rule: No larger disk on a smaller one."
        ])
        
        # Load assets
        rods = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg").set_color(WHITE) for _ in range(3)])
        disks = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg").set_color("#FF4500") for _ in range(3)])
        
        # Scale disks to different sizes
        for i, disk in enumerate(disks):
            disk.scale(0.8 - i * 0.1)

        # Layout rods as requested by Critic
        self.place_at_grid(rods[0], 'D2', scale_factor=1.5)
        self.place_at_grid(rods[1], 'D4', scale_factor=1.5)
        self.place_at_grid(rods[2], 'D6', scale_factor=1.5)
        self.add(rods)
        
        # Arrange disks on the first rod (D2)
        disks.arrange(UP, buff=0.05)
        disks.next_to(rods[0], UP, buff=-0.5)
        self.add(disks)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        # Move smallest disk to rod[1]
        self.play(disks[2].animate.move_to(rods[1].get_top() + UP * 0.1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        # Show illegal move attempt (red color)
        illegal_move = disks[1].copy().set_color("#FF0000")
        self.play(illegal_move.animate.move_to(rods[1].get_top() + UP * 0.3))
        self.wait(2)
        self.remove(illegal_move)
