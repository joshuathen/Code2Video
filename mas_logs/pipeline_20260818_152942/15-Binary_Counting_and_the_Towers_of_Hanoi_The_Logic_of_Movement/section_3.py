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
        lecture_lines = ["Binary counting maps to disk moves.", "Changing bits dictate which disk moves.", "Count 1 is move disk 1."]
        self.setup_layout("Connecting Moves to Binary Counting", lecture_lines)
        
        # Assets
        disk_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        peg_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg"
        
        # Objects (loading SVGs as icons)
        disk_obj = SVGMobject(disk_svg)
        peg_obj = SVGMobject(peg_svg)
        
        # Binary representations
        binary_display = VGroup(
            Text("001", color=BLUE),
            Text("010", color=GREEN),
            Text("011", color=YELLOW)
        ).arrange(DOWN, buff=0.5)
        
        # Applying requested layout fixes
        self.place_in_area(binary_display, 'B4', 'E6', scale_factor=0.7)
        
        # Disk objects
        disk1 = SVGMobject(disk_svg).set_color(RED)
        disk2 = SVGMobject(disk_svg).set_color(BLUE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(FadeIn(binary_display))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_at_grid(disk2, 'B3', scale_factor=0.6)
        self.play(FadeIn(disk2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.place_at_grid(disk1, 'C3', scale_factor=0.6)
        self.play(FadeIn(disk1))
        self.wait(1)
