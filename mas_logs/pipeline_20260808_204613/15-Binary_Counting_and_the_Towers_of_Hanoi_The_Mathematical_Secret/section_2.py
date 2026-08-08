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
        self.setup_layout("The Towers of Hanoi Puzzle", [
            "Move disks from start to end peg.",
            "Never place larger disks on smaller ones.",
            "Move one disk at a time only."
        ])

        # Assets
        peg_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg")
        disk_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")

        # Setup pegs
        peg_a = peg_img.copy()
        peg_b = peg_img.copy()
        peg_c = peg_img.copy()
        pegs = VGroup(peg_a, peg_b, peg_c).arrange(RIGHT, buff=1.5)
        
        # Setup disks
        disks = VGroup(*[disk_img.copy().scale(0.5 - i*0.1) for i in range(3)])
        
        # Assemble
        full_setup = VGroup(pegs, disks)
        self.place_in_area(full_setup, "A4", "F6", scale_factor=0.6)
        self.add(full_setup)

        # Positioning disks
        for i, disk in enumerate(disks):
            disk.next_to(pegs[0], UP, buff=0.1 + i*0.2)
            disk.align_to(pegs[0], LEFT)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(RED))
        
        # Highlight smallest disk
        smallest_disk = disks[0]
        self.play(smallest_disk.animate.set_color("#FF4500"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        # Move smallest disk from A to C
        target_pos = pegs[2].get_center() + UP * 0.1
        self.play(smallest_disk.animate.move_to(target_pos))
