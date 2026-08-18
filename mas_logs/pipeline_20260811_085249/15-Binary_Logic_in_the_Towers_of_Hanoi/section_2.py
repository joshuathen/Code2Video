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
        self.setup_layout("Mapping Towers of Hanoi to Binary", [
            "Towers of Hanoi puzzles have specific moves.",
            "Binary counting maps directly to these moves.",
            "Three disks take seven total moves."
        ])
        
        # Assets
        peg_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg"
        disk_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"

        # === Animation for Lecture Line 1 ===
        # Display three pegs labeled Peg A, B, C using SVG assets
        peg_a = SVGMobject(peg_path, color=WHITE)
        label_a = Text("A", font_size=20).next_to(peg_a, DOWN)
        peg_b = SVGMobject(peg_path, color=WHITE)
        label_b = Text("B", font_size=20).next_to(peg_b, DOWN)
        peg_c = SVGMobject(peg_path, color=WHITE)
        label_c = Text("C", font_size=20).next_to(peg_c, DOWN)
        
        pegs = VGroup(VGroup(peg_a, label_a), VGroup(peg_b, label_b), VGroup(peg_c, label_c)).arrange(RIGHT, buff=1.0)
        
        self.place_in_area(pegs, 'D1', 'F6', scale_factor=0.6)
        self.play(FadeIn(pegs))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Add 3 disks on Peg A (#1E90FF)
        disk1 = SVGMobject(disk_path, color="#1E90FF")
        disk2 = SVGMobject(disk_path, color="#1E90FF")
        disk3 = SVGMobject(disk_path, color="#1E90FF")
        disks = VGroup(disk3, disk2, disk1).arrange(UP, buff=0.1).move_to(peg_a.get_bottom() + UP*0.3)
        
        self.play(FadeIn(disks))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Highlight disks with labels 1, 2, 3
        t1 = Text("1", font_size=24, color=WHITE).move_to(disk1.get_center())
        t2 = Text("2", font_size=24, color=WHITE).move_to(disk2.get_center())
        t3 = Text("3", font_size=24, color=WHITE).move_to(disk3.get_center())
        labels = VGroup(t3, t2, t1)
        
        diagram_group = VGroup(pegs, disks, labels)
        
        self.place_at_grid(labels, 'B3', scale_factor=0.7)
        self.place_in_area(diagram_group, 'C1', 'E3', scale_factor=0.5)
        
        self.play(Write(labels))
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
