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
        self.setup_layout("Introduction to Towers of Hanoi", ["Towers of Hanoi has three rods.", "Move disks following strict size rules.", "Goal is moving the full stack."])
        
        # Load assets
        rod_img = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg"
        disk_img = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        
        # Create rods and disks using SVG Assets
        rods = VGroup(*[SVGMobject(rod_img, color=BLUE_D).set_fill(BLUE_D, opacity=1) for _ in range(3)])
        
        # Position rods in area C2-F5
        rod_positions = ['C2', 'C4', 'C6']
        for i, pos in enumerate(rod_positions):
            self.place_at_grid(rods[i], pos, scale_factor=0.5)

        disk_sizes = [0.8, 1.0, 1.2]
        disks = VGroup(*[SVGMobject(disk_img, color="#E74C3C").set_fill("#E74C3C", opacity=1) for s in disk_sizes])
        
        # Initial stack on rod A (C2)
        for i, disk in enumerate(disks):
            self.place_at_grid(disk, 'C2')
            # Adjust scaling based on size
            disk.scale(0.3 + 0.1 * i)
            disk.shift(DOWN * (0.4 * (2 - i)))

        self.add(rods, disks)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        # Animate moving the smallest disk (disks[2]) to middle peg (C4)
        self.play(disks[2].animate.move_to(self.grid['C4'] + DOWN * 0.4))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        # Animate moving the medium disk (disks[1]) to destination peg (C6)
        self.play(disks[1].animate.move_to(self.grid['C6'] + DOWN * 0.4))
