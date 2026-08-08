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
        self.setup_layout("Mapping Moves to Binary", [
            "- Hanoi moves match the binary sequence.",
            "- Three disks require seven total moves.",
            "- Binary count zero-zero-one is move one.",
            "- Binary count one-one-one is move seven.",
            "- The sequence maps binary to disk movement."
        ])
        
        # Elements
        move_label = Text("Move:", font_size=32)
        self.place_at_grid(move_label, 'B3', scale_factor=0.7)
        
        binary_display = Text("000", font_size=64, color="#00CED1")
        self.place_at_grid(binary_display, 'C3', scale_factor=0.8)
        
        disk = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg")
        self.place_at_grid(disk, 'C4', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00CED1"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00CED1"))
        new_val = Text("001", font_size=64, color="#00CED1")
        self.place_at_grid(new_val, 'C4', scale_factor=0.8)
        self.play(Transform(binary_display, new_val), disk.animate.shift(RIGHT))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF4500"))
        new_val_2 = Text("111", font_size=64, color="#FF4500")
        self.place_at_grid(new_val_2, 'D4', scale_factor=0.8)
        self.play(Transform(binary_display, new_val_2), disk.animate.shift(LEFT))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#ADFF2F"))
        self.wait(1)
